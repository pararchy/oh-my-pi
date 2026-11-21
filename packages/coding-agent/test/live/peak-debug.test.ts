import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { LocalLiveSessionController, type LiveSpeechSink, type LocalLiveDeps } from "@oh-my-pi/pi-coding-agent/live/local-controller";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";

const LOUD = new Float32Array(320).fill(0.2);
const MERGE_MS = 900;

function harness() {
	const streams: any[] = [];
	const prompts: string[] = [];
	const state = { streaming: false };
	let onAudio: (a: Float32Array) => void = () => {};
	const session: any = {
		settings: Settings.isolated({ "live.localVoice": "harper" }),
		modelRegistry: {},
		sessionId: "x",
		get isStreaming() { return state.streaming; },
		async prompt(t: string) { prompts.push(t); return true; },
		async abort() {},
		async sendCustomMessage() { return false; },
		subscribe() { return () => {}; },
	};
	const sink: any = { speaking: false, clears: 0, pushDelta: () => {}, flush: () => {}, clear() { this.speaking = false; }, isSpeaking() { return this.speaking; }, setModelSource: () => {}, setEnhancer: () => {} };
	const controller = new LocalLiveSessionController({
		session,
		extractAssistantText: () => "",
		callbacks: { onPhase: () => {}, onLevels: () => {}, onTranscript: () => {}, onTerminal: () => {} },
		deps: {
			segmentMergeMs: MERGE_MS,
			startStream(o: any) { const s = { o, audio: [] as Float32Array[] }; streams.push(s); return { pushAudio: (p: Float32Array) => s.audio.push(p.slice()), pushDetect: () => {}, cancel: () => {}, stop: async () => "" }; },
			startCapture(cb: any) { onAudio = cb; return { stop: () => {} }; },
			createVocalizer: () => sink,
		},
	});
	return { controller, streams, prompts, segment: (t: string) => streams.at(-1)!.o.onSegment(t), audio: onAudio };
}
describe("dbg", () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());
	it("stop then loud thank-you", async () => {
		const h = harness();
		await h.controller.start();
		h.segment("stop");
		vi.advanceTimersByTime(MERGE_MS + 10); await Promise.resolve();
		for (let n = 0; n < 20; n++) h.audio(LOUD);
		console.log("peak after loud:", (h.controller as any)["#utterancePeak"]);
		h.segment("Thank you.");
		vi.advanceTimersByTime(MERGE_MS + 10); await Promise.resolve();
		console.log("prompts:", h.prompts);
		await h.controller.stop();
	});
});
