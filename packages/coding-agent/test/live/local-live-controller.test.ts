import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	LocalLiveSessionController,
	type LocalLiveDeps,
	type LiveSpeechSink,
} from "@oh-my-pi/pi-coding-agent/live/local-controller";
import type { LiveSessionCallbacks } from "@oh-my-pi/pi-coding-agent/live/controller";
import type { SttStreamHandle } from "@oh-my-pi/pi-coding-agent/stt/asr-client";

const MERGE_MS = 40;

interface FakeSession {
	prompts: { text: string; behavior: string | undefined }[];
	aborts: string[];
	custom: unknown[];
	streaming: boolean;
	session: unknown;
}

function fakeSession(streaming = false): FakeSession {
	const state: FakeSession = {
		prompts: [],
		aborts: [],
		custom: [],
		streaming,
		session: undefined,
	};
	const settings = Settings.isolated({ "live.localVoice": "af_kore" });
	state.session = {
		settings,
		sessionId: "test-session",
		agent: undefined,
		modelRegistry: {},
		get isStreaming() {
			return state.streaming;
		},
		async prompt(text: string, options?: { streamingBehavior?: string }) {
			state.prompts.push({ text, behavior: options?.streamingBehavior });
			return true;
		},
		async abort(options?: { reason?: string }) {
			state.aborts.push(options?.reason ?? "");
		},
		async sendCustomMessage(message: unknown) {
			state.custom.push(message);
			return false;
		},
		subscribe() {
			return () => {};
		},
	};
	return state;
}

interface Harness {
	controller: LocalLiveSessionController;
	fake: FakeSession;
	pushAudio: (samples: Float32Array) => void;
	segment: (text: string) => void;
	streamPushed: Float32Array[];
	sink: LiveSpeechSink & { cleared: number; speaking: boolean };
	phases: string[];
}

function createHarness(streaming = false): Harness {
	const fake = fakeSession(streaming);
	const phases: string[] = [];
	const callbacks: LiveSessionCallbacks = {
		onPhase: phase => phases.push(phase),
		onLevels: () => {},
		onTranscript: () => {},
		onTerminal: () => {},
	};
	let onSegment: (text: string) => void = () => {};
	let onAudio: (samples: Float32Array) => void = () => {};
	const streamPushed: Float32Array[] = [];
	const sink: Harness["sink"] = {
		cleared: 0,
		speaking: false,
		pushDelta: () => {},
		flush: () => {},
		clear() {
			this.cleared += 1;
			this.speaking = false;
		},
		isSpeaking() {
			return this.speaking;
		},
		setModelSource: () => {},
		setEnhancer: () => {},
	};
	const stream: SttStreamHandle = {
		pushAudio: samples => streamPushed.push(samples),
		stop: () => Promise.resolve(""),
		cancel: () => {},
	};
	const deps: LocalLiveDeps = {
		segmentMergeMs: MERGE_MS,
		startStream(options) {
			onSegment = options.onSegment;
			return stream;
		},
		startCapture(callback) {
			onAudio = callback;
			return { stop: () => {} };
		},
		createVocalizer: () => sink,
	};
	const controller = new LocalLiveSessionController({
		session: fake.session as never,
		callbacks,
		extractAssistantText: () => "",
		deps,
	});
	return {
		controller,
		fake,
		pushAudio: samples => onAudio(samples),
		segment: text => onSegment(text),
		streamPushed,
		sink,
		phases,
	};
}

const LOUD = new Float32Array(320).fill(0.5);
const QUIET = new Float32Array(320).fill(0.001);

/** Fire the merge timer and let the queued async work land. */
async function settle(): Promise<void> {
	vi.advanceTimersByTime(MERGE_MS + 1);
	for (let i = 0; i < 8; i++) await Promise.resolve();
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("LocalLiveSessionController", () => {
	it("merges segments across short pauses into one prompt", async () => {
		const h = createHarness();
		await h.controller.start();
		h.segment("first part");
		h.segment("second part");
		await settle();
		expect(h.fake.prompts).toEqual([{ text: "first part second part", behavior: undefined }]);
		await h.controller.stop();
	});

	it("steers the running turn instead of aborting while the agent streams", async () => {
		const h = createHarness(true);
		await h.controller.start();
		h.segment("add this too");
		await settle();
		expect(h.fake.prompts[0]?.behavior).toBe("steer");
		expect(h.fake.aborts).toEqual([]);
		await h.controller.stop();
	});

	it("gates the mic while speech plays and barge-in clears the vocalizer", async () => {
		const h = createHarness();
		await h.controller.start();
		h.sink.speaking = true;
		h.pushAudio(QUIET);
		expect(h.streamPushed.at(-1)?.every(s => s === 0)).toBe(true);
		for (let i = 0; i < 6; i++) h.pushAudio(LOUD);
		expect(h.sink.cleared).toBeGreaterThan(0);
		// Buffered loud chunks flushed through after barge-in.
		expect(h.streamPushed.at(-1)).toBe(LOUD);
		await h.controller.stop();
	});

	it("spoken 'stop' interrupts work without starting a new turn", async () => {
		const h = createHarness(true);
		await h.controller.start();
		h.segment("stop");
		await settle();
		expect(h.fake.aborts.length).toBe(1);
		expect(h.fake.prompts).toEqual([]);
		await h.controller.stop();
	});

	it("hides the voice-mode note from the transcript but keeps it in context", async () => {
		const h = createHarness();
		await h.controller.start();
		const note = h.fake.custom[0] as { display: boolean; content: string };
		expect(note.display).toBe(false);
		expect(note.content).toContain("spoken");
		await h.controller.stop();
	});
});
