/**
 * Local speech recognition and Kokoro playback around the session's configured
 * reasoning model (which may be remote). Only finalized STT segments become
 * prompts; previews remain volatile. A finalized utterance during work aborts
 * the in-flight turn and becomes a fresh turn — the voice equivalent of Esc
 * then typing — so a voice user is never made to wait for a long answer or a
 * long tool batch. Echo gating is an energy heuristic, not AEC.
 */
import type { AgentSessionEvent } from "../session/agent-session-events";
import type { AgentSession } from "../session/agent-session";
import { AudioCapture } from "@oh-my-pi/pi-natives";
import { toError } from "@oh-my-pi/pi-utils";
import type { LivePhase } from "@oh-my-pi/pi-tui/apps/live-visualizer";
import { resolveRoleChain } from "../config/model-resolver";
import { roleCandidatePool } from "../config/model-roles";
import { DEFAULT_STT_MODEL_KEY, resolveSttModelSpec, type SttModelKey } from "../stt/models";
import { isSttModelCached, downloadSttModel } from "../stt/downloader";
import { sttClient, type SttStreamHandle } from "../stt/asr-client";
import { cfgSttLanguage } from "../stt/settings";
import { isTtsModelCached, downloadTtsModel } from "../tts/downloader";
import { resolveLocalSpeechModelId, Vocalizer } from "../tts/vocalizer";
import { TTS_LOCAL_VOICE_VALUES } from "../tts/models";
import { SpeechEnhancer } from "../tts/speech-enhancer";
import { cfgLiveLocalVoice, cfgLiveMicThreshold, cfgLivePersonality } from "./settings";
import { LiveInputGate } from "./input-gate";
import type { LiveSessionControllerOptions, LiveTranscript } from "./controller";

/** Quiet span after a committed segment before the utterance submits as a turn. */
const SEGMENT_MERGE_MS = 900;
/** RMS floor that must be sustained to count as a barge-in while speech plays. */
const BARGE_IN_LEVEL = 0.05;
/** Barge-in duration is measured in PCM samples, independent of callback size. */
const BARGE_IN_SAMPLES = 4_000;
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
	startStream(options: {
		onPartial(text: string): void;
		onSegment(text: string): void;
		language?: string;
		speechDetection?: "energy" | "silero";
		onSpeech?(active: boolean): void;
		onError?(error: Error): void;
		onDetector?(detector: "energy" | "silero"): void;
	}): SttStreamHandle;
	startCapture(onAudio: (samples: Float32Array) => void): { stop(): void };
	createVocalizer(): LiveSpeechSink;
	/** Utterance merge window; tests shrink it. Defaults to {@link SEGMENT_MERGE_MS}. */
	segmentMergeMs?: number;
}

export class LocalLiveSessionController {
	readonly #deps: LocalLiveDeps | undefined;
	readonly #session: AgentSession;
	readonly #callbacks: LiveSessionControllerOptions["callbacks"];
	readonly #inputGate: LiveInputGate;
	readonly #micThreshold: number;
	readonly #startupAbort = new AbortController();
	#modelKey: SttModelKey | undefined;
	#streamGeneration = 0;
	#hasUtterance = false;
	#awaitingFinal = false;
	#speechInterrupted = false;
	/** Latest detector state; true while the VAD holds a speech run open. */
	#vadSpeechActive = false;
	/** Resolved detector for the active stream; `"energy"` disables VAD gating. */
	#detector: "energy" | "silero" = "energy";
	#silence = new Float32Array(0);

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
	#bargeSamples = 0;
	#speakReleaseAt = 0;

	constructor(options: LiveSessionControllerOptions & { deps?: LocalLiveDeps }) {
		this.#session = options.session;
		this.#deps = options.deps;
		this.#callbacks = options.callbacks;
		this.#micThreshold = cfgLiveMicThreshold.get(this.#session.settings);
		this.#inputGate = new LiveInputGate({ threshold: this.#micThreshold });
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
			this.#modelKey = modelKey;
			if (!this.#deps && !(await isSttModelCached(modelKey))) {
				await downloadSttModel(modelKey, undefined, { signal: this.#startupAbort.signal });
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
					if (!(await downloadTtsModel(ttsModelKey, undefined, this.#startupAbort.signal))) {
						throw new Error("Local speech synthesis model setup failed.");
					}
				}
			}
			// Warm the neural speech detector so its ~2 MB model is cached before
			// the first call. Failure is non-fatal: barge-in falls back to energy.
			if (!this.#deps) {
				await sttClient.prepareVad(modelKey, { signal: this.#startupAbort.signal }).catch(() => {});
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
			if (this.#stopped) throw this.#failure ?? new Error("The live session stopped while starting.");

			this.#unsubscribeSession = this.#session.subscribe(event =>
				this.#guardEvent(() => this.#handleSessionEvent(event)),
			);

			if (!this.#muted) this.#openStream();

			if (this.#deps) {
				const capture = this.#deps.startCapture(samples =>
					this.#guardEvent(() => this.#handleMicrophoneAudio(samples)),
				);
				if (this.#stopped) capture.stop();
				else this.#captureHandle = capture;
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
			if (this.#stopped) throw this.#failure ?? new Error("The live session stopped while starting.");
			this.#phaseTicker = setInterval(() => this.#guardEvent(() => this.#refreshPhase()), PHASE_TICK_MS);
			this.#phaseTicker.unref?.();
			this.#refreshPhase();
		} catch (cause) {
			const error = toError(cause);
			if (!this.#stopped) this.#reportFailure(error);
			await this.stop();
			throw error;
		}
	}

	#openStream(): void {
		const modelKey = this.#modelKey;
		if (!modelKey || this.#stopped || this.#muted) return;
		const generation = ++this.#streamGeneration;
		this.#vadSpeechActive = false;
		this.#detector = "energy";
		const options = {
			language: cfgSttLanguage.get(this.#session.settings) || undefined,
			speechDetection: "silero" as const,
			onDetector: (detector: "energy" | "silero") => {
				if (generation === this.#streamGeneration && !this.#muted) {
					this.#guardEvent(() => {
						this.#detector = detector;
					});
				}
			},
			onSpeech: (active: boolean) => {
				if (generation === this.#streamGeneration && !this.#muted) {
					this.#guardEvent(() => this.#handleSpeechActivity(active));
				}
			},
			onError: (error: Error) => {
				if (generation === this.#streamGeneration && !this.#muted) {
					this.#guardEvent(() => this.#reportFailure(error));
				}
			},
			onPartial: (text: string) => {
				if (generation === this.#streamGeneration && !this.#muted) {
					this.#guardEvent(() => this.#handlePartial(text));
				}
			},
			onSegment: (text: string) => {
				if (generation === this.#streamGeneration && !this.#muted) {
					this.#guardEvent(() => this.#handleSegment(text));
				}
			},
		};
		const stream = this.#deps ? this.#deps.startStream(options) : sttClient.startStream(modelKey, options);
		if (this.#stopped || this.#muted || generation !== this.#streamGeneration) stream.cancel();
		else this.#stream = stream;
	}

	#resetInput(): void {
		this.#clearSubmitTimer();
		this.#utteranceParts = [];
		this.#partial = "";
		this.#hasUtterance = false;
		this.#awaitingFinal = false;
		this.#inputGate.reset();
		this.#bargeSamples = 0;
		this.#bargeBuffer = [];
		this.#cancelInterruptedSubmit();
	}

	toggleMute(): void {
		if (this.#stopped) return;
		this.#muted = !this.#muted;
		this.#guardEvent(() => {
			if (this.#muted) {
				this.#streamGeneration += 1;
				const stream = this.#stream;
				this.#stream = undefined;
				this.#resetInput();
				stream?.cancel();
				this.#emitTranscript(undefined);
				this.#emitLevels(0, this.#outputLevel());
			} else {
				this.#openStream();
			}
		});
		this.#refreshPhase();
	}

	stop(): Promise<void> {
		if (!this.#stopPromise) this.#stopPromise = this.#stop();
		return this.#stopPromise;
	}

	async #stop(): Promise<void> {
		this.#stopped = true;
		this.#startupAbort.abort();
		this.#streamGeneration += 1;
		this.#resetInput();
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
		if (this.#stopped || this.#muted || !this.#stream || samples.length === 0) return;
		const level = microphoneLevel(samples);
		const speaking = this.#vocalizer?.isSpeaking() ?? false;
		if (speaking) this.#speakReleaseAt = Date.now() + ECHO_RELEASE_MS;
		this.#emitLevels(level, speaking ? 0.4 : 0);
		// The detector hears every mic frame, including playback; the energy gate
		// alone decides what reaches the recognizer.
		this.#stream?.pushDetect(samples);
		if (!speaking && Date.now() >= this.#speakReleaseAt) {
			this.#bargeSamples = 0;
			this.#bargeBuffer = [];
			this.#inputGate.push(samples, this.#pushRecognizedAudio);
			return;
		}

		// Playback exclusion: energy prefilters sustained loud input; the neural
		// detector confirms speech before playback is interrupted (see
		// #handleSpeechActivity). Preserve the onset in owned PCM.
		this.#inputGate.reset();
		if (level >= BARGE_IN_LEVEL) {
			const buffered = samples.slice(0, BARGE_IN_SAMPLES);
			this.#bargeBuffer.push(buffered);
			this.#bargeSamples += buffered.length;
			// VAD confirmation is required only when the neural detector resolved;
			// on fallback the energy heuristic keeps its original behavior.
			if (this.#bargeSamples >= BARGE_IN_SAMPLES && (this.#detector !== "silero" || this.#vadSpeechActive)) {
				this.#vocalizer?.clear();
				this.#speechInterrupted = true;
				this.#speakReleaseAt = 0;
				for (const chunk of this.#bargeBuffer) this.#inputGate.push(chunk, this.#pushRecognizedAudio);
				this.#bargeBuffer = [];
				this.#bargeSamples = 0;
				return;
			}
		} else {
			this.#bargeSamples = 0;
			this.#bargeBuffer = [];
		}
		if (this.#silence.length !== samples.length) this.#silence = new Float32Array(samples.length);
		this.#stream?.pushAudio(this.#silence);
	}

	#handleSpeechActivity(active: boolean): void {
		this.#vadSpeechActive = active;
		if (!active) return;
		const speaking = (this.#vocalizer?.isSpeaking() ?? false) || Date.now() < this.#speakReleaseAt;
		if (speaking && this.#bargeSamples >= BARGE_IN_SAMPLES) {
			this.#vocalizer?.clear();
			this.#speechInterrupted = true;
			this.#speakReleaseAt = 0;
			for (const chunk of this.#bargeBuffer) this.#inputGate.push(chunk, this.#pushRecognizedAudio);
			this.#bargeBuffer = [];
			this.#bargeSamples = 0;
		}
	}

	/** Stable callback: IPC consumes emitted frames synchronously. */
	readonly #pushRecognizedAudio = (audio: Float32Array): void => {
		if (this.#stopped || this.#muted || !this.#stream) return;
		if (this.#inputGate.active && microphoneLevel(audio) >= this.#micThreshold) {
			this.#awaitingFinal = true;
			this.#clearSubmitTimer();
		}
		this.#stream.pushAudio(audio);
	};

	// --- Transcription into turns --------------------------------------------

	#handlePartial(text: string): void {
		if (this.#stopped || this.#muted) return;
		const partial = text.trim();
		if (partial) {
			if (!this.#hasUtterance) {
				this.#hasUtterance = true;
				this.#userTurn += 1;
			}
			this.#awaitingFinal = true;
			this.#clearSubmitTimer();
		}
		this.#partial = partial;
		this.#emitUserTranscript(false);
	}

	#handleSegment(text: string): void {
		if (this.#stopped || this.#muted) return;
		const segment = text.trim();
		this.#partial = "";
		this.#awaitingFinal = false;
		if (segment) {
			if (!this.#hasUtterance) {
				this.#hasUtterance = true;
				this.#userTurn += 1;
			}
			this.#utteranceParts.push(segment);
			this.#emitUserTranscript(false);
		}
		if (this.#utteranceParts.length > 0) this.#armSubmit();
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
		if (this.#partial || this.#awaitingFinal) return;
		if (this.#inputGate.active) {
			this.#armSubmit();
			return;
		}
		const text = this.#utteranceParts.join(" ").trim();
		if (!text) return;
		this.#emitUserTranscript(true);
		this.#utteranceParts = [];
		this.#partial = "";
		this.#hasUtterance = false;
		this.#clearSubmitTimer();
		this.#emitTranscript(undefined);
		if (this.#stopped) return;
		const working = this.#session.isStreaming;
		const speaking = (this.#vocalizer?.isSpeaking() ?? false) || this.#speechInterrupted;
		const stopRequested = STOP_PATTERN.test(text);
		if (working || speaking) {
			this.#vocalizer?.clear();
			this.#speechInterrupted = true;
		}
		if (working) {
			// Interrupt first — Esc then typing: abort the in-flight turn, then
			// deliver the utterance once streaming clears. A bare "stop" has no
			// payload to re-prompt; it only cancels.
			void this.#session.abort({ reason: "user spoke during a voice turn" }).catch(cause => {
				if (!this.#stopped) this.#reportFailure(toError(cause));
			});
			if (!stopRequested) this.#armInterruptedSubmit(text);
			return;
		}
		if (stopRequested && speaking) return;
		this.#promptUtterance(text);
		this.#refreshPhase();
	}

	#promptUtterance(text: string): void {
		// Never serialize behind this promise: it may resolve only at agent_end.
		void this.#session
			.prompt(text, {
				userInitiated: true,
				streamingBehavior: "steer",
			})
			.catch(cause => {
				if (!this.#stopped) this.#reportFailure(toError(cause));
			});
	}

	/** Utterance captured while a turn was still unwinding: submit once
	 * streaming clears; controller stop or a newer utterance cancels it. */
	#interruptedUtterance: { text: string; timer?: NodeJS.Timeout } | undefined;

	#armInterruptedSubmit(text: string): void {
		this.#cancelInterruptedSubmit();
		const entry: { text: string; timer?: NodeJS.Timeout } = { text };
		this.#interruptedUtterance = entry;
		const trySubmit = () => {
			if (this.#stopped || this.#interruptedUtterance !== entry) return;
			if (this.#session.isStreaming) {
				entry.timer = setTimeout(trySubmit, 100);
				entry.timer.unref?.();
				return;
			}
			this.#interruptedUtterance = undefined;
			this.#promptUtterance(entry.text);
			this.#refreshPhase();
		};
		entry.timer = setTimeout(trySubmit, 100);
		entry.timer.unref?.();
	}

	#cancelInterruptedSubmit(): void {
		if (this.#interruptedUtterance?.timer) clearTimeout(this.#interruptedUtterance.timer);
		this.#interruptedUtterance = undefined;
	}

	// --- Session events → speech and phase -----------------------------------

	#handleSessionEvent(event: AgentSessionEvent): void {
		switch (event.type) {
			case "message_start":
				if (event.message.role === "user") this.#vocalizer?.clear();
				if (event.message.role === "assistant") this.#speechInterrupted = false;
				break;
			case "message_update":
				if (!this.#speechInterrupted && event.assistantMessageEvent.type === "text_delta") {
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
