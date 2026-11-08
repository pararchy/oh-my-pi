/**
 * Local voice engine for `/live` — mic → local STT → the session's own turn →
 * local Kokoro TTS. Unlike the Codex realtime path there is no remote model: the
 * SAME AgentSession answers, so the persona, tools and transcript are shared with
 * text mode. A spoken prompt while the agent is streaming steers the running
 * turn; tool work never aborts on user speech.
 *
 * Echo: while speech is playing the mic is gated unless input stays clearly loud
 * for {@link BARGE_IN_STREAK} chunks (headphones make this irrelevant; on
 * speakers it is a deliberate talk-over). Utterances that outlive a
 * {@link SEGMENT_MERGE_MS} pause are submitted as one turn so mid-thought pauses
 * do not split prompts.
 */
import type { AgentSessionEvent } from "../session/agent-session-events";
import type { AgentSession } from "../session/agent-session";
import { AudioCapture } from "@oh-my-pi/pi-natives";
import { logger, toError } from "@oh-my-pi/pi-utils";
import type { LivePhase } from "@oh-my-pi/pi-tui/apps/live-visualizer";
import { resolveRoleChain } from "../config/model-resolver";
import { roleCandidatePool } from "../config/model-roles";
import { DEFAULT_STT_MODEL_KEY, resolveSttModelSpec, type SttModelKey } from "../stt/models";
import { isSttModelCached, downloadSttModel } from "../stt/downloader";
import { sttClient, type SttStreamHandle } from "../stt/asr-client";
import { isTtsModelCached, downloadTtsModel } from "../tts/downloader";
import { resolveLocalSpeechModelId, Vocalizer } from "../tts/vocalizer";
import { TTS_LOCAL_VOICE_VALUES } from "../tts/models";
import { SpeechEnhancer } from "../tts/speech-enhancer";
import { cfgLiveLocalVoice, cfgLivePersonality } from "./settings";
import type { LiveSessionControllerOptions, LiveTranscript } from "./controller";

/** Quiet span after a committed segment before the utterance submits as a turn. */
const SEGMENT_MERGE_MS = 900;
/** RMS floor that must be sustained to count as a barge-in while speech plays. */
const BARGE_IN_LEVEL = 0.05;
/** Consecutive ~64 ms chunks above the floor before barge-in trips (~250 ms). */
const BARGE_IN_STREAK = 4;
/** Quiet span after playback ends before mic gating drops (~echo ring-down). */
const ECHO_RELEASE_MS = 350;
/** Assistant-speech phase poll; isSpeaking() is cheap so this can run fast. */
const PHASE_TICK_MS = 150;
/** Sample rate the STT worker and AudioCapture share. */
const SAMPLE_RATE = 16_000;
/** Spoken commands that interrupt work in flight. */
const STOP_PATTERN = /^\s*(?:stop|cancel|cancel that|nevermind|never mind|shut up)[.!…]*\s*$/i;

const VOICE_MODE_NOTE = `This conversation runs over a live voice channel: the user's words arrive as speech-to-text (transcription quirks expected) and your replies are spoken aloud by text-to-speech. Answer briefly and conversationally in plain spoken sentences — no markdown, lists, code blocks, or tables unless the user explicitly asks for them; offer to put detail in the text chat instead. Do not mention this note or the voice system.`;

function clampLevel(level: number): number {
	if (!Number.isFinite(level) || level <= 0) return 0;
	return Math.min(1, level);
}

function microphoneLevel(samples: Float32Array): number {
	if (samples.length === 0) return 0;
	let sumSquares = 0;
	for (let index = 0; index < samples.length; index += 1) {
		const sample = samples[index] ?? 0;
		sumSquares += sample * sample;
	}
	return clampLevel(Math.sqrt(sumSquares / samples.length));
}

/** The slice of {@link Vocalizer} the controller drives once speaking starts. */
export type LiveSpeechSink = Pick<
	Vocalizer,
	"pushDelta" | "flush" | "clear" | "isSpeaking" | "setModelSource" | "setEnhancer"
>;

/** Injectable seams for tests: capture chunks, STT segments and speech playback. */
export interface LocalLiveDeps {
	startStream(options: { onPartial(text: string): void; onSegment(text: string): void }): SttStreamHandle;
	startCapture(onAudio: (samples: Float32Array) => void): { stop(): void };
	createVocalizer(): LiveSpeechSink;
	/** Utterance merge window; tests shrink it. Defaults to {@link SEGMENT_MERGE_MS}. */
	segmentMergeMs?: number;
}

export class LocalLiveSessionController {
	readonly #deps: LocalLiveDeps | undefined;
	readonly #session: AgentSession;
	readonly #callbacks: LiveSessionControllerOptions["callbacks"];

	#recorder: AudioCapture | undefined;
	#captureHandle: { stop(): void } | undefined;
	#stream: SttStreamHandle | undefined;
	#vocalizer: LiveSpeechSink | undefined;
	#unsubscribeSession: (() => void) | undefined;
	#stopPromise: Promise<void> | undefined;
	#started = false;
	#stopped = false;
	#terminalEmitted = false;
	#failure: Error | undefined;
	#muted = false;
	#phase: LivePhase = "connecting";
	#userTurn = 0;
	/** Segments of the in-progress utterance, merged across short pauses. */
	#utteranceParts: string[] = [];
	/** Volatile in-progress segment text reported by the STT stream. */
	#partial = "";
	#submitTimer: NodeJS.Timeout | undefined;
	#phaseTicker: NodeJS.Timeout | undefined;
	/** Chunks kept while speech plays so a barge-in keeps its first words. */
	#bargeBuffer: Float32Array[] = [];
	#bargeStreak = 0;
	#speakReleaseAt = 0;
	#emitQueue: Promise<void> = Promise.resolve();

	constructor(options: LiveSessionControllerOptions & { deps?: LocalLiveDeps }) {
		this.#session = options.session;
		this.#deps = options.deps;
		this.#callbacks = options.callbacks;
	}

	get phase(): LivePhase {
		return this.#phase;
	}

	get muted(): boolean {
		return this.#muted;
	}

	get sdpAnswer(): string | undefined {
		return undefined;
	}

	async start(): Promise<void> {
		if (this.#stopped) {
			throw (
				this.#failure ?? new Error("This live session has already stopped; create a new controller to reconnect.")
			);
		}
		if (this.#started) return;
		this.#started = true;
		this.#emitPhase("connecting", true);
		this.#emitTranscript(undefined);

		try {
			const modelKey = this.#resolveSttModelKey();
			if (!this.#deps && !(await isSttModelCached(modelKey))) {
				await downloadSttModel(modelKey, undefined);
			}
			if (this.#stopped) throw this.#failure ?? new Error("The live session stopped while starting.");

			// Speech voice: a live-session override wins, then the global speech voice.
			const localVoice = cfgLiveLocalVoice.get(this.#session.settings).trim();
			const speechSettings = this.#session.settings.overlay({
				"speech.enabled": true,
				...(localVoice && TTS_LOCAL_VOICE_VALUES.includes(localVoice) ? { "speech.voice": localVoice } : {}),
			});
			const vocalizer = this.#deps ? this.#deps.createVocalizer() : new Vocalizer();
			vocalizer.setModelSource({ settings: speechSettings, registry: this.#session.modelRegistry });
			if (this.#session.agent && !this.#deps) {
				vocalizer.setEnhancer(
					new SpeechEnhancer({
						settings: this.#session.settings,
						registry: this.#session.modelRegistry,
						sessionId: this.#session.sessionId,
						metadataResolver: provider => this.#session.agent.metadataForProvider(provider),
					}),
				);
			}
			this.#vocalizer = vocalizer;
			if (!this.#deps) {
				const ttsModelKey = resolveLocalSpeechModelId({
					settings: speechSettings,
					registry: this.#session.modelRegistry,
				});
				if (!(await isTtsModelCached(ttsModelKey))) {
					await downloadTtsModel(ttsModelKey);
				}
			}
			if (this.#stopped) throw this.#failure ?? new Error("The live session stopped while starting.");

			// Hidden per-session voice note: spoken-style replies, plus the persona's
			// spoken personality when configured. Stored once, drives every later turn.
			const personality = cfgLivePersonality.get(this.#session.settings).trim();
			await this.#session.sendCustomMessage({
				customType: "live_voice_note",
				content: personality ? `${VOICE_MODE_NOTE}\n\n${personality}` : VOICE_MODE_NOTE,
				display: false,
				attribution: "user",
			});

			this.#unsubscribeSession = this.#session.subscribe(event =>
				this.#guardEvent(() => this.#handleSessionEvent(event)),
			);

			this.#stream = this.#deps
				? this.#deps.startStream({
						onPartial: text => this.#guardEvent(() => this.#handlePartial(text)),
						onSegment: text => this.#guardEvent(() => this.#handleSegment(text)),
					})
				: sttClient.startStream(modelKey, {
						onPartial: text => this.#guardEvent(() => this.#handlePartial(text)),
						onSegment: text => this.#guardEvent(() => this.#handleSegment(text)),
					});

			if (this.#deps) {
				this.#captureHandle = this.#deps.startCapture(samples =>
					this.#guardEvent(() => this.#handleMicrophoneAudio(samples)),
				);
			} else {
				const recorder = new AudioCapture(SAMPLE_RATE, (error, samples) => {
					if (error) {
						this.#reportFailure(error);
						return;
					}
					this.#guardEvent(() => this.#handleMicrophoneAudio(samples));
				});
				if (this.#stopped) {
					try {
						recorder.stop();
					} catch {
						// Preserve the failure that stopped startup.
					}
					throw this.#failure ?? new Error("The live session stopped while recording began.");
				}
				this.#recorder = recorder;
			}
			this.#phaseTicker = setInterval(() => this.#guardEvent(() => this.#refreshPhase()), PHASE_TICK_MS);
			this.#phaseTicker.unref?.();
			this.#refreshPhase();
		} catch (cause) {
			const error = toError(cause);
			this.#reportFailure(error);
			await this.stop();
			throw error;
		}
	}

	toggleMute(): void {
		if (this.#stopped) return;
		this.#muted = !this.#muted;
		if (this.#muted) {
			this.#utteranceParts = [];
			this.#partial = "";
			this.#clearSubmitTimer();
			this.#emitTranscript(undefined);
			this.#emitLevels(0, this.#outputLevel());
		}
		this.#refreshPhase();
	}

	stop(): Promise<void> {
		if (!this.#stopPromise) this.#stopPromise = this.#stop();
		return this.#stopPromise;
	}

	async #stop(): Promise<void> {
		this.#stopped = true;
		this.#clearSubmitTimer();
		if (this.#phaseTicker) {
			clearInterval(this.#phaseTicker);
			this.#phaseTicker = undefined;
		}
		this.#unsubscribeSession?.();
		this.#unsubscribeSession = undefined;
		let cleanupError: Error | undefined;
		const stream = this.#stream;
		this.#stream = undefined;
		if (stream) {
			try {
				stream.cancel();
			} catch (cause) {
				cleanupError = toError(cause);
			}
		}
		const recorder = this.#recorder;
		this.#recorder = undefined;
		if (recorder) {
			try {
				recorder.stop();
			} catch (cause) {
				cleanupError ??= toError(cause);
			}
		}
		const captureHandle = this.#captureHandle;
		this.#captureHandle = undefined;
		if (captureHandle) {
			try {
				captureHandle.stop();
			} catch (cause) {
				cleanupError ??= toError(cause);
			}
		}
		this.#vocalizer?.clear();
		this.#vocalizer = undefined;
		if (cleanupError) this.#emitPhaseSafely("error");
		this.#emitTerminal(cleanupError);
	}

	// --- Microphone and echo gate -------------------------------------------

	#handleMicrophoneAudio(samples: Float32Array): void {
		if (this.#stopped) return;
		const stream = this.#stream;
		if (!stream) return;
		const level = microphoneLevel(samples);
		const speaking = this.#vocalizer?.isSpeaking() ?? false;
		if (speaking) this.#speakReleaseAt = Date.now() + ECHO_RELEASE_MS;

		if (this.#muted) {
			this.#emitLevels(0, speaking ? 0.4 : 0);
			return;
		}
		this.#emitLevels(level, speaking ? 0.4 : 0);

		const gated = speaking || Date.now() < this.#speakReleaseAt;
		if (!gated) {
			this.#bargeStreak = 0;
			stream.pushAudio(samples);
			return;
		}

		// While speech plays, input only reaches the STT as a sustained loud run —
		// speaker bleed sits below the floor unless the user talks over the agent.
		if (level >= BARGE_IN_LEVEL) {
			this.#bargeBuffer.push(samples);
			if (this.#bargeBuffer.length > BARGE_IN_STREAK * 2) this.#bargeBuffer.shift();
			this.#bargeStreak += 1;
			if (this.#bargeStreak === BARGE_IN_STREAK) {
				this.#vocalizer?.clear();
				this.#speakReleaseAt = 0;
				for (const buffered of this.#bargeBuffer) stream.pushAudio(buffered);
				this.#bargeBuffer = [];
			} else {
				stream.pushAudio(new Float32Array(samples.length));
			}
			return;
		}
		this.#bargeStreak = 0;
		this.#bargeBuffer = [];
		stream.pushAudio(new Float32Array(samples.length));
	}

	// --- Transcription into turns --------------------------------------------

	#handlePartial(text: string): void {
		if (this.#stopped || this.#muted) return;
		if (!this.#partial.trim() && this.#utteranceParts.length === 0) this.#userTurn += 1;
		this.#partial = text;
		this.#emitUserTranscript(false);
	}

	#handleSegment(text: string): void {
		if (this.#stopped || this.#muted) return;
		const segment = text.trim();
		if (segment) {
			if (!this.#partial.trim() && this.#utteranceParts.length === 0) this.#userTurn += 1;
			this.#utteranceParts.push(segment);
			this.#emitUserTranscript(false);
			this.#armSubmit();
		}
	}

	#armSubmit(): void {
		this.#clearSubmitTimer();
		this.#submitTimer = setTimeout(() => {
			this.#submitTimer = undefined;
			this.#guardEvent(() => this.#submitUtterance());
		}, this.#deps?.segmentMergeMs ?? SEGMENT_MERGE_MS);
		this.#submitTimer.unref?.();
	}

	#clearSubmitTimer(): void {
		if (this.#submitTimer) {
			clearTimeout(this.#submitTimer);
			this.#submitTimer = undefined;
		}
	}

	#submitUtterance(): void {
		const parts = [...this.#utteranceParts];
		if (this.#partial.trim()) parts.push(this.#partial.trim());
		const text = parts.join(" ").trim();
		this.#utteranceParts = [];
		this.#partial = "";
		this.#clearSubmitTimer();
		this.#emitUserTranscript(true);
		this.#emitTranscript(undefined);
		if (!text) return;
		if (STOP_PATTERN.test(text)) {
			this.#vocalizer?.clear();
			if (this.#session.isStreaming) {
				this.#queueWork(() => this.#session.abort({ reason: "user requested stop via voice" }));
			}
			return;
		}
		this.#vocalizer?.clear();
		this.#queueWork(async () => {
			const started = await this.#session.prompt(text, {
				userInitiated: true,
				streamingBehavior: this.#session.isStreaming ? "steer" : undefined,
			});
			if (!started) {
				logger.warn("local live: spoken prompt was dropped", { text });
			}
		});
		this.#refreshPhase();
	}

	/** Serialize session mutations so turns never overlap a stop. */
	#queueWork(work: () => Promise<unknown>): void {
		this.#emitQueue = this.#emitQueue
			.then(async () => {
				if (!this.#stopped) await work();
			})
			.catch(cause => this.#reportFailure(toError(cause)));
	}

	// --- Session events → speech and phase -----------------------------------

	#handleSessionEvent(event: AgentSessionEvent): void {
		switch (event.type) {
			case "message_start":
				if (event.message.role === "user") this.#vocalizer?.clear();
				break;
			case "message_update":
				if (event.assistantMessageEvent.type === "text_delta") {
					this.#vocalizer?.pushDelta(event.assistantMessageEvent.delta);
				}
				break;
			case "agent_end":
				this.#vocalizer?.flush();
				break;
			default:
				break;
		}
		this.#refreshPhase();
	}

	#outputLevel(): number {
		return this.#vocalizer?.isSpeaking() ? 0.4 : 0;
	}

	#refreshPhase(): void {
		if (this.#stopped) return;
		if (this.#muted) {
			this.#emitPhase("muted");
		} else if (this.#vocalizer?.isSpeaking()) {
			this.#emitPhase("speaking");
		} else if (this.#session.isStreaming) {
			this.#emitPhase("working");
		} else {
			this.#emitPhase("listening");
		}
	}

	// --- Emitters ------------------------------------------------------------

	#emitUserTranscript(final: boolean): void {
		const parts = [...this.#utteranceParts];
		if (this.#partial.trim()) parts.push(this.#partial.trim());
		const text = parts.join(" ").trim();
		if (!text) return;
		this.#emitTranscript({ role: "user", turn: this.#userTurn, text, final });
	}

	#emitPhase(phase: LivePhase, force = false): void {
		if (!force && this.#phase === phase) return;
		this.#phase = phase;
		try {
			this.#callbacks.onPhase(phase);
		} catch (cause) {
			this.#reportFailure(toError(cause));
		}
	}

	#emitPhaseSafely(phase: LivePhase): void {
		this.#phase = phase;
		try {
			this.#callbacks.onPhase(phase);
		} catch {
			// Terminal callback is the final error boundary for UI failures.
		}
	}

	#emittedInput = -1;
	#emittedOutput = -1;
	#emitLevels(input: number, output: number): void {
		if (input === this.#emittedInput && output === this.#emittedOutput) return;
		try {
			this.#callbacks.onLevels(input, output);
			this.#emittedInput = input;
			this.#emittedOutput = output;
		} catch (cause) {
			this.#reportFailure(toError(cause));
		}
	}

	#emitTranscript(transcript: LiveTranscript | undefined): void {
		try {
			this.#callbacks.onTranscript(transcript);
		} catch (cause) {
			this.#reportFailure(toError(cause));
		}
	}

	#reportFailure(error: Error): void {
		if (this.#terminalEmitted) return;
		this.#failure = error;
		this.#emitPhaseSafely("error");
		this.#emitTerminal(error);
		void this.stop();
	}

	#emitTerminal(error?: Error): void {
		if (this.#terminalEmitted) return;
		this.#terminalEmitted = true;
		try {
			this.#callbacks.onTerminal(error);
		} catch {
			// Nothing remains above the terminal callback to receive its error.
		}
	}

	#guardEvent(handler: () => void): void {
		if (this.#stopped) return;
		try {
			handler();
		} catch (cause) {
			this.#reportFailure(toError(cause));
		}
	}

	/** Dictation-role model when it is local; otherwise the built-in default. */
	#resolveSttModelKey(): SttModelKey {
		try {
			const pool = roleCandidatePool("dictation", this.#session.settings, this.#session.modelRegistry);
			const candidate = resolveRoleChain("dictation", this.#session.settings, pool).find(
				entry => entry.model.api === "local-inference",
			);
			return resolveSttModelSpec(candidate?.model.id).key;
		} catch {
			return resolveSttModelSpec(DEFAULT_STT_MODEL_KEY).key;
		}
	}
}
