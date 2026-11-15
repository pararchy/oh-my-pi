import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import {
	LocalLiveSessionController,
	type LiveSpeechSink,
	type LocalLiveDeps,
} from "@oh-my-pi/pi-coding-agent/live/local-controller";
import type { LiveTranscript } from "@oh-my-pi/pi-coding-agent/live/controller";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import type { SttStreamHandle } from "@oh-my-pi/pi-coding-agent/stt/asr-client";

const MERGE_MS = 900;
const controllers: LocalLiveSessionController[] = [];
const LOUD = new Float32Array(320).fill(0.2);
const QUIET = new Float32Array(320).fill(0.003);

function harness() {
	const streams: {
		options: Parameters<LocalLiveDeps["startStream"]>[0];
		audio: Float32Array[];
		cancelled: boolean;
	}[] = [];
	const prompts: string[] = [];
	const transcripts: LiveTranscript[] = [];
	const errors: Error[] = [];
	const pendingPrompt = Promise.withResolvers<boolean>();
	const noteReady = Promise.withResolvers<boolean>();
	const state = {
		streaming: false,
		holdPrompt: false,
		holdNote: false,
		aborts: 0,
		captureStopped: false,
		captures: 0,
		subscriptions: 0,
	};
	const sink: LiveSpeechSink & { speaking: boolean; clears: number } = {
		speaking: false,
		clears: 0,
		pushDelta: () => {},
		flush: () => {},
		clear() {
			this.clears++;
			this.speaking = false;
		},
		isSpeaking() {
			return this.speaking;
		},
		setModelSource: () => {},
		setEnhancer: () => {},
	};
	let onAudio: (audio: Float32Array) => void = () => {};
	const session = {
		settings: Settings.isolated({ "live.localVoice": "af_kore" }),
		modelRegistry: {},
		sessionId: "test-voice",
		get isStreaming() {
			return state.streaming;
		},
		async prompt(text: string) {
			prompts.push(text);
			if (state.streaming) return false;
			if (state.holdPrompt) {
				state.streaming = true;
				return pendingPrompt.promise;
			}
			return true;
		},
		async abort() {
			state.aborts++;
			state.streaming = false;
		},
		async sendCustomMessage() {
			if (state.holdNote) return noteReady.promise;
			return false;
		},
		subscribe() {
			state.subscriptions++;
			return () => {
				state.subscriptions--;
			};
		},
	} as unknown as AgentSession; // Deliberate in-process fake: only the controller's session surface is exercised.
	const controller = new LocalLiveSessionController({
		session,
		extractAssistantText: () => "",
		callbacks: {
			onPhase: () => {},
			onLevels: () => {},
			onTranscript: t => {
				if (t) transcripts.push(t);
			},
			onTerminal: e => {
				if (e) errors.push(e);
			},
		},
		deps: {
			segmentMergeMs: MERGE_MS,
			startStream(options) {
				const stream = { options, audio: [] as Float32Array[], detect: [] as Float32Array[], cancelled: false };
				streams.push(stream);
				return {
					pushAudio: pcm => stream.audio.push(pcm.slice()),
					pushDetect: pcm => stream.detect.push(pcm.slice()),
					cancel: () => {
						stream.cancelled = true;
					},
					stop: async () => "",
				} satisfies SttStreamHandle;
			},
			startCapture(callback) {
				state.captures++;
				onAudio = callback;
				return {
					stop: () => {
						state.captureStopped = true;
					},
				};
			},
			createVocalizer: () => sink,
		},
	});
	controllers.push(controller);
	return {
		controller,
		streams,
		prompts,
		transcripts,
		errors,
		pendingPrompt,
		noteReady,
		state,
		sink,
		partial: (text: string) => streams.at(-1)!.options.onPartial(text),
		segment: (text: string) => streams.at(-1)!.options.onSegment(text),
		audio: (samples: Float32Array) => onAudio(samples),
	};
}

async function advance(ms = MERGE_MS + 1) {
	vi.advanceTimersByTime(ms);
	for (let n = 0; n < 8; n++) await Promise.resolve();
}
beforeEach(() => vi.useFakeTimers());
afterEach(async () => {
	for (const controller of controllers.splice(0)) await controller.stop();
	vi.useRealTimers();
});

describe("local live conversation", () => {
	it("replaces a partial with final words once and emits one stable final transcript", async () => {
		const h = harness();
		await h.controller.start();
		h.partial("move the lamp");
		h.partial("move the lantern");
		h.segment("Move the lantern.");
		await advance();
		expect(h.prompts).toEqual(["Move the lantern."]);
		expect(h.transcripts.filter(t => t.final)).toEqual([
			{ role: "user", turn: 1, final: true, text: "Move the lantern." },
		]);
		expect(h.transcripts.every(t => t.turn === 1)).toBe(true);
		h.segment("Now turn left.");
		await advance();
		expect(h.transcripts.at(-1)?.turn).toBe(2);
	});
	it("never submits an unfinished continuation, then merges its finalized text", async () => {
		const h = harness();
		await h.controller.start();
		h.segment("Please move");
		await advance(300);
		h.partial("the green");
		await advance(MERGE_MS * 3);
		expect(h.prompts).toEqual([]);
		h.segment("the green lantern.");
		await advance();
		expect(h.prompts).toEqual(["Please move the green lantern."]);
	});
	it("does not turn a standalone partial into a prompt", async () => {
		const h = harness();
		await h.controller.start();
		h.partial("not finished");
		await advance(MERGE_MS * 5);
		expect(h.prompts).toEqual([]);
	});
	it("interrupts an active turn: abort first, then the utterance becomes a fresh turn", async () => {
		const h = harness();
		h.state.holdPrompt = true;
		await h.controller.start();
		h.segment("Start the report.");
		await advance();
		h.segment("Include the latest figures.");
		await advance();
		// The second utterance aborts the in-flight turn immediately…
		expect(h.state.aborts).toBe(1);
		expect(h.prompts).toEqual(["Start the report."]);
		// …and is re-submitted once streaming clears (Esc then typing).
		h.pendingPrompt.resolve(true);
		h.state.streaming = false;
		await advance(200);
		expect(h.prompts).toEqual(["Start the report.", "Include the latest figures."]);
		expect(h.errors).toEqual([]);
		await h.controller.stop();
		expect(h.state.captureStopped).toBe(true);
	});
	it("spoken stop aborts without waiting for the active answer to resolve", async () => {
		const h = harness();
		h.state.holdPrompt = true;
		await h.controller.start();
		h.segment("Start a long task.");
		await advance();
		h.segment("stop");
		await advance();
		expect(h.state.aborts).toBe(1);
		expect(h.prompts).toEqual(["Start a long task."]);
		h.pendingPrompt.resolve(true);
	});
	it("does not silently swallow stop or thank you when idle", async () => {
		const h = harness();
		await h.controller.start();
		h.segment("stop");
		await advance();
		// "Thank you." matches the silence-hallucination list, so it needs real
		// admitted loudness to reach the session — Whisper's phantom version does.
		for (let n = 0; n < 20; n++) h.audio(LOUD);
		h.segment("Thank you.");
		// Keep the gate's frame-counted release from closing before the merge
		// timer fires — submit waits while the gate is active.
		for (let n = 0; n < 40; n++) h.audio(QUIET);
		await advance();
		expect(h.prompts).toEqual(["stop", "Thank you."]);
	});
	it("mute discards stale recognizer callbacks, pre-mute words and audio", async () => {
		const h = harness();
		await h.controller.start();
		const old = h.streams[0]!;
		h.partial("discard this");
		h.controller.toggleMute();
		expect(old.cancelled).toBe(true);
		h.audio(LOUD);
		h.controller.toggleMute();
		old.options.onPartial("stale preview");
		old.options.onSegment("stale final");
		await advance();
		expect(h.prompts).toEqual([]);
		h.segment("Only these words.");
		await advance();
		expect(h.prompts).toEqual(["Only these words."]);
		expect(h.transcripts.some(t => t.text.includes("stale"))).toBe(false);
	});
	it("rejects quiet room audio and short clicks before recognition", async () => {
		const h = harness();
		await h.controller.start();
		for (let n = 0; n < 100; n++) h.audio(QUIET);
		h.audio(LOUD);
		for (let n = 0; n < 100; n++) h.audio(QUIET);
		expect(h.streams[0]!.audio.every(pcm => pcm.every(value => value === 0))).toBe(true);
	});
	it("drops classic whisper hallucinations that arrive sub-speech, keeps real quiet speech", async () => {
		const h = harness();
		await h.controller.start();
		// Quiet room audio never opens the gate; STT still emits the phantom line.
		for (let n = 0; n < 20; n++) h.audio(QUIET);
		h.segment("Thank you for watching!");
		await advance();
		expect(h.prompts).toEqual([]);
		// Quiet but genuinely requested content still delivers.
		h.segment("Add the lamp to the scene.");
		await advance();
		expect(h.prompts).toEqual(["Add the lamp to the scene."]);
	});
	it("requires 250ms of loud audio to barge in, not four small callbacks", async () => {
		const h = harness();
		await h.controller.start();
		h.sink.speaking = true;
		for (let n = 0; n < 12; n++) h.audio(LOUD);
		expect(h.sink.clears).toBe(0);
		h.audio(LOUD);
		expect(h.sink.clears).toBe(1);
		expect(h.streams[0]!.audio.some(pcm => pcm.some(value => value > 0.1))).toBe(true);
	});
	it("waits for detector confirmation before interrupting playback when silero resolves", async () => {
		const h = harness();
		await h.controller.start();
		h.sink.speaking = true;
		h.streams[0]!.options.onDetector?.("silero");
		for (let n = 0; n < 13; n++) h.audio(LOUD);
		// Loud mic alone no longer clears playback once a neural detector is live.
		expect(h.sink.clears).toBe(0);
		h.streams[0]!.options.onSpeech?.(true);
		expect(h.sink.clears).toBe(1);
		expect(h.streams[0]!.audio.some(pcm => pcm.some(value => value > 0.1))).toBe(true);
	});
	it("stopping during asynchronous setup cannot reopen capture or subscriptions", async () => {
		const h = harness();
		h.state.holdNote = true;
		const start = h.controller.start();
		await Promise.resolve();
		await h.controller.stop();
		h.noteReady.resolve(false);
		await expect(start).rejects.toThrow("stopped");
		expect(h.state.captures).toBe(0);
		expect(h.state.subscriptions).toBe(0);
		expect(h.streams).toEqual([]);
	});
	it("discards finalized and preview words received after call teardown", async () => {
		const h = harness();
		await h.controller.start();
		const old = h.streams[0]!;
		await h.controller.stop();
		old.options.onPartial("late");
		old.options.onSegment("late final");
		await advance();
		expect(h.prompts).toEqual([]);
		expect(h.transcripts).toEqual([]);
	});
});
