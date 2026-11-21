import { createHash } from "node:crypto";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { getTinyModelsCacheDir } from "@oh-my-pi/pi-utils";
import type {
	AutomaticSpeechRecognitionOutput,
	AutomaticSpeechRecognitionPipeline,
	ProgressInfo,
} from "../subprocess/transformers-types";
import {
	errorMessage,
	errorText,
	loadTransformersRuntime,
	MemoizedRuntime,
	replayCachedReady,
	sendLog,
	sendProgress,
	TRANSFORMERS_VERSION,
} from "../subprocess/worker-runtime";
import { resolveTinyModelDevicePreference, type TinyModelDevice, tinyModelDeviceLoadOrder } from "../tiny/device";
import { resolveTinyModelDtypeOverride, type TinyModelDtype } from "../tiny/dtype";
import { runFramedStdioWorker } from "../daemon/modules/speech-frames";
import type { SttTransport, SttWorkerInbound, SttWorkerOutbound } from "./asr-protocol";
import { type EndpointerEvent, StreamEndpointer } from "./endpointer";
import {
	getSttModelSpec,
	type SherpaSttModelSpec,
	type SttModel,
	type SttModelKey,
	type TransformersSttModelSpec,
} from "./models";
import {
	loadSherpaRuntime,
	SHERPA_PACKAGE,
	SHERPA_VERSION,
	type SherpaOfflineRecognizer,
	type SherpaOnlineRecognizer,
	type SherpaOnlineStream,
	type SherpaRuntime,
	type SherpaVad,
} from "./sherpa-runtime";

const ASR_TASK = "automatic-speech-recognition";
// Whisper long-form decoding: split into 30s windows with 5s overlap so audio of
// any length transcribes without exceeding the 30s receptive field.
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;
// The client always resamples to 16 kHz mono float32 before sending; sherpa-onnx
// is told the true input rate (it resamples internally to its feature config).
const ASR_SAMPLE_RATE = 16_000;
const ONLINE_TERMINAL_PADDING = new Float32Array(ASR_SAMPLE_RATE * 0.6);
// Hub origin for raw sherpa-onnx model files (encoder/decoder/joiner/tokens).
const HF_RESOLVE_BASE = "https://huggingface.co";
// Coalesce download progress so streaming a multi-hundred-MB model file doesn't
// flood the IPC channel with one event per chunk.
const PROGRESS_EMIT_BYTES = 4_000_000;
// Silero VAD v5 model weights, fetched raw from the snakers4/silero-vad repo
// (the HuggingFace mirror is gated). The upstream file path is unversioned, so
// the digest below is verified after every download — a changed upstream file
// fails loudly instead of silently loading different weights.
const VAD_MODEL_FILENAME = "silero_vad.onnx";
const VAD_MODEL_URL = `https://raw.githubusercontent.com/snakers4/silero-vad/v5.0/files/${VAD_MODEL_FILENAME}`;
const VAD_MODEL_SHA256 = "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396";
// Seconds of audio retained inside the native detector so finalized segments
// can be drained; far above `maxSpeechDuration`.
const VAD_BUFFER_SECONDS = 60;

const sttModelDevicePreference = resolveTinyModelDevicePreference();
const sttModelDtypeOverride = resolveTinyModelDtypeOverride();

/**
 * Subset of the transformers.js ASR call options we set. The index signature
 * mirrors `GenerationFunctionParameters` so this is assignable to the pipeline's
 * `Partial<AutomaticSpeechRecognitionConfig>` param (not re-exported from the
 * package root, so we model only what we pass).
 */
interface AsrCallOptions {
	chunk_length_s: number;
	stride_length_s: number;
	return_timestamps: boolean;
	task?: string;
	language?: string;
	[key: string]: unknown;
}

interface TransformersRuntime {
	env: {
		cacheDir?: string;
		allowLocalModels?: boolean;
		logLevel?: unknown;
	};
	LogLevel: {
		ERROR: unknown;
	};
	pipeline: (
		task: typeof ASR_TASK,
		model: string,
		options: {
			device: TinyModelDevice;
			dtype: TinyModelDtype;
			progress_callback: (info: ProgressInfo) => void;
		},
	) => Promise<AutomaticSpeechRecognitionPipeline>;
}

/** A warm model plus the engine that loaded it; cached per tier key. */
type LoadedModel =
	| { engine: "transformers"; pipeline: AutomaticSpeechRecognitionPipeline }
	| { engine: "sherpa"; mode: "offline"; recognizer: SherpaOfflineRecognizer }
	| { engine: "sherpa"; mode: "online"; recognizer: SherpaOnlineRecognizer };

const models = new Map<SttModelKey, Promise<LoadedModel>>();
// Serialize all model inference on a single chain: the recognizers are not
// guaranteed reentrant and there is one CPU-bound model per tier. Batch
// transcribes and live-stream segment/partial decodes share this lock.
let modelLock = Promise.resolve();
function runOnModel<T>(work: () => Promise<T>): Promise<T> {
	const run = modelLock.then(work, work);
	modelLock = run.then(
		() => undefined,
		() => undefined,
	);
	return run;
}
const transformersRuntime = new MemoizedRuntime<TransformersRuntime>();
const sherpaRuntime = new MemoizedRuntime<SherpaRuntime>();

function getSttRuntimeDir(): string {
	const key = TRANSFORMERS_VERSION.replace(/[^A-Za-z0-9._-]/g, "_");
	return path.join(path.dirname(getTinyModelsCacheDir()), "stt-runtime", `transformers-${key}`);
}

async function loadPipelineOnDevice(
	transformers: TransformersRuntime,
	spec: TransformersSttModelSpec,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
	device: TinyModelDevice,
): Promise<AutomaticSpeechRecognitionPipeline> {
	return transformers.pipeline(ASR_TASK, spec.repo, {
		device,
		dtype: sttModelDtypeOverride ?? spec.dtype,
		progress_callback: info => sendProgress(transport, requestId, modelKey, info),
	});
}

async function loadPipelineWithDeviceFallback(
	transformers: TransformersRuntime,
	spec: TransformersSttModelSpec,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
): Promise<{ pipeline: AutomaticSpeechRecognitionPipeline; device: TinyModelDevice }> {
	const devices = tinyModelDeviceLoadOrder(sttModelDevicePreference);
	if (devices[0] !== sttModelDevicePreference.device) {
		sendLog(transport, "warn", "stt: requested device is unsafe in the worker; using CPU", {
			modelKey,
			repo: spec.repo,
			requestedDevice: sttModelDevicePreference.device,
			device: devices[0],
		});
	}
	for (let i = 0; i < devices.length; i += 1) {
		const device = devices[i]!;
		try {
			return {
				pipeline: await loadPipelineOnDevice(transformers, spec, modelKey, transport, requestId, device),
				device,
			};
		} catch (error) {
			if (i === devices.length - 1) throw error;
			const fallbackDevice = devices[i + 1]!;
			sendLog(transport, "warn", "stt: accelerated device failed; falling back", {
				modelKey,
				repo: spec.repo,
				device,
				fallbackDevice,
				error: errorMessage(error),
			});
		}
	}
	throw new Error("No stt model devices configured");
}

async function loadTransformersModel(
	spec: TransformersSttModelSpec,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
): Promise<LoadedModel> {
	const transformers = await loadTransformersRuntime(
		transformersRuntime,
		transport,
		requestId,
		modelKey,
		getSttRuntimeDir,
	);
	const startedAt = performance.now();
	const { pipeline, device } = await loadPipelineWithDeviceFallback(
		transformers,
		spec,
		modelKey,
		transport,
		requestId,
	);
	sendLog(transport, "debug", "stt: local model loaded", {
		modelKey,
		repo: spec.repo,
		engine: "transformers",
		device,
		requestedDevice: sttModelDevicePreference.device,
		dtype: sttModelDtypeOverride ?? spec.dtype,
		elapsedMs: Math.round(performance.now() - startedAt),
	});
	return { engine: "transformers", pipeline };
}

/**
 * Stream a single sherpa-onnx model file from the Hub into the cache, writing to
 * a `.part` sidecar and renaming on completion so an interrupted fetch never
 * reads as cached. Emits coalesced per-file progress for the aggregating client.
 */
async function downloadSherpaFile(
	repo: string,
	filename: string,
	dest: string,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
): Promise<void> {
	const url = `${HF_RESOLVE_BASE}/${repo}/resolve/main/${filename}`;
	await downloadModelFile(url, dest, modelKey, transport, requestId, `${repo}/${filename}`, filename);
}

/**
 * Stream a model file into the cache, writing to a `.part` sidecar and renaming
 * on completion so an interrupted fetch never reads as cached. Emits coalesced
 * progress for the aggregating client; when `sha256` is given the renamed file
 * is verified and deleted on mismatch.
 */
async function downloadModelFile(
	url: string,
	dest: string,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
	name: string,
	filename: string,
	sha256?: string,
): Promise<void> {
	const response = await fetch(url, { redirect: "follow" });
	if (!response.ok || !response.body) {
		throw new Error(`Failed to download ${filename} (${name}): HTTP ${response.status}`);
	}
	const total = Number(response.headers.get("content-length") ?? 0);
	transport.send({
		type: "progress",
		id: requestId,
		event: { modelKey, status: "download", name, file: filename },
	});
	const part = `${dest}.part`;
	const handle = await fs.open(part, "w");
	let loaded = 0;
	let lastEmitted = 0;
	const reader = response.body.getReader();
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			if (!value) continue;
			await handle.write(value);
			loaded += value.byteLength;
			if (loaded - lastEmitted >= PROGRESS_EMIT_BYTES || (total > 0 && loaded >= total)) {
				lastEmitted = loaded;
				transport.send({
					type: "progress",
					id: requestId,
					event: { modelKey, status: "progress", name, file: filename, loaded, total: total || loaded },
				});
			}
		}
	} finally {
		await handle.close();
	}
	await fs.rename(part, dest);
	if (!sha256) return;
	const digest = createHash("sha256")
		.update(await fs.readFile(dest))
		.digest("hex");
	if (digest !== sha256) {
		await fs.unlink(dest).catch(() => {});
		throw new Error(`Downloaded ${filename} failed sha256 verification (${digest})`);
	}
}

/**
 * Ensure the pinned Silero VAD model is present in the cache. The `.part`
 * rename guarantees a full file came through {@link downloadModelFile}, whose
 * sha256 verification covers integrity; a wrong-size leftover is re-fetched.
 */
async function ensureVadModelFile(modelKey: SttModelKey, transport: SttTransport, requestId: string): Promise<string> {
	const dir = path.join(getTinyModelsCacheDir(), "snakers4/silero-vad");
	await fs.mkdir(dir, { recursive: true });
	const dest = path.join(dir, VAD_MODEL_FILENAME);
	const present = await fs
		.stat(dest)
		.then(stats => stats.size > 0)
		.catch(() => false);
	if (!present) {
		await downloadModelFile(
			VAD_MODEL_URL,
			dest,
			modelKey,
			transport,
			requestId,
			`snakers4/silero-vad/${VAD_MODEL_FILENAME}`,
			VAD_MODEL_FILENAME,
			VAD_MODEL_SHA256,
		);
	}
	return dest;
}

/**
 * Load the native detector for one session. Model file verification and the
 * shared runtime install are reused across sessions via the module caches.
 */
async function createVad(modelKey: SttModelKey, transport: SttTransport, requestId: string): Promise<SherpaVad> {
	const [runtime, modelPath] = await Promise.all([
		sherpaRuntime.load(() =>
			loadSherpaRuntime(phase =>
				transport.send({
					type: "progress",
					id: requestId,
					event: { modelKey, status: phase, name: `${SHERPA_PACKAGE}@${SHERPA_VERSION}` },
				}),
			),
		),
		ensureVadModelFile(modelKey, transport, requestId),
	]);
	return new runtime.Vad(
		{
			sileroVad: {
				model: modelPath,
				threshold: 0.5,
				minSpeechDuration: 0.25,
				minSilenceDuration: 0.5,
				maxSpeechDuration: 15,
				windowSize: 512,
			},
			sampleRate: ASR_SAMPLE_RATE,
			numThreads: 1,
			provider: "cpu",
			debug: 0,
		},
		VAD_BUFFER_SECONDS,
	);
}

/**
 * Ensure all sherpa-onnx model files for a tier are present in the cache,
 * downloading any that are missing, and return their absolute paths.
 */
async function ensureSherpaModelFiles(
	spec: SherpaSttModelSpec,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
): Promise<{ encoder: string; decoder: string; joiner: string; tokens: string }> {
	const dir = path.join(getTinyModelsCacheDir(), spec.repo);
	await fs.mkdir(dir, { recursive: true });
	const resolved = {} as { encoder: string; decoder: string; joiner: string; tokens: string };
	for (const role in spec.files) {
		const key = role as keyof typeof spec.files;
		const filename = spec.files[key];
		const dest = path.join(dir, filename);
		const present = await fs
			.stat(dest)
			.then(stats => stats.size > 0)
			.catch(() => false);
		if (!present) await downloadSherpaFile(spec.repo, filename, dest, modelKey, transport, requestId);
		resolved[key] = dest;
	}
	return resolved;
}

async function loadSherpaModel(
	spec: SherpaSttModelSpec,
	modelKey: SttModelKey,
	transport: SttTransport,
	requestId: string,
): Promise<LoadedModel> {
	const runtime = await sherpaRuntime.load(() =>
		loadSherpaRuntime(phase =>
			transport.send({
				type: "progress",
				id: requestId,
				event: { modelKey, status: phase, name: `${SHERPA_PACKAGE}@${SHERPA_VERSION}` },
			}),
		),
	);
	const files = await ensureSherpaModelFiles(spec, modelKey, transport, requestId);
	const startedAt = performance.now();
	const numThreads = Math.max(1, Math.min(4, os.availableParallelism()));
	const config = {
		modelConfig: {
			transducer: { encoder: files.encoder, decoder: files.decoder, joiner: files.joiner },
			tokens: files.tokens,
			modelType: spec.modelType,
			numThreads,
			provider: "cpu",
			debug: 0,
		},
		decodingMethod: "greedy_search",
	};
	const model: LoadedModel =
		spec.mode === "online"
			? {
					engine: "sherpa",
					mode: "online",
					recognizer: new runtime.OnlineRecognizer({ ...config, enableEndpoint: true }),
				}
			: { engine: "sherpa", mode: "offline", recognizer: await runtime.OfflineRecognizer.createAsync(config) };
	sendLog(transport, "debug", "stt: local model loaded", {
		modelKey,
		repo: spec.repo,
		engine: "sherpa",
		provider: "cpu",
		numThreads,
		elapsedMs: Math.round(performance.now() - startedAt),
	});
	return model;
}

async function loadModel(modelKey: SttModelKey, transport: SttTransport, requestId: string): Promise<LoadedModel> {
	const spec = getSttModelSpec(modelKey);
	if (!spec) throw new Error(`Unknown stt model: ${modelKey}`);
	const cached = replayCachedReady(models, modelKey, transport, requestId, ASR_TASK, spec.repo);
	if (cached) return cached;

	const loading =
		spec.engine === "sherpa"
			? loadSherpaModel(spec, modelKey, transport, requestId)
			: loadTransformersModel(spec, modelKey, transport, requestId);
	const loaded = loading.then(
		model => {
			transport.send({
				type: "progress",
				id: requestId,
				event: { modelKey, status: "ready", task: ASR_TASK, model: spec.repo },
			});
			return model;
		},
		error => {
			models.delete(modelKey);
			throw error;
		},
	);
	models.set(modelKey, loaded);
	return loaded;
}

async function decodeSegment(
	model: LoadedModel,
	spec: SttModel,
	audio: Float32Array,
	language: string | undefined,
): Promise<string> {
	if (model.engine === "sherpa") {
		if (model.mode === "online") {
			const stream = model.recognizer.createStream();
			stream.acceptWaveform({ samples: audio, sampleRate: ASR_SAMPLE_RATE });
			stream.acceptWaveform({ samples: ONLINE_TERMINAL_PADDING, sampleRate: ASR_SAMPLE_RATE });
			stream.inputFinished();
			while (model.recognizer.isReady(stream)) model.recognizer.decode(stream);
			return model.recognizer.getResult(stream).text.trim();
		}
		const stream = model.recognizer.createStream();
		stream.acceptWaveform({ samples: audio, sampleRate: ASR_SAMPLE_RATE });
		const result = await model.recognizer.decodeAsync(stream);
		return (result.text ?? "").trim();
	}
	const options: AsrCallOptions = {
		chunk_length_s: CHUNK_LENGTH_S,
		stride_length_s: STRIDE_LENGTH_S,
		return_timestamps: false,
	};
	// English-only Whisper checkpoints reject `language`/`task`; multilingual ones
	// take the configured source language (auto-detected when omitted).
	if (!spec.englishOnly) {
		options.task = "transcribe";
		if (language) options.language = language;
	}
	const output = (await model.pipeline(audio, options)) as AutomaticSpeechRecognitionOutput;
	return (output.text ?? "").trim();
}

async function transcribeAudio(
	transport: SttTransport,
	requestId: string,
	modelKey: SttModelKey,
	audio: Float32Array,
	language: string | undefined,
): Promise<string> {
	const spec = getSttModelSpec(modelKey);
	if (!spec) throw new Error(`Unknown stt model: ${modelKey}`);
	const model = await loadModel(modelKey, transport, requestId);
	return runOnModel(() => decodeSegment(model, spec, audio, language));
}

async function handleBatchRequest(
	transport: SttTransport,
	request: Extract<SttWorkerInbound, { type: "transcribe" | "download" | "vad_prepare" }>,
): Promise<void> {
	try {
		if (request.type === "download") {
			await loadModel(request.modelKey, transport, request.id);
			transport.send({ type: "downloaded", id: request.id });
			return;
		}
		if (request.type === "vad_prepare") {
			await createVad(request.modelKey, transport, request.id);
			transport.send({ type: "vad_ready", id: request.id });
			return;
		}
		const text = await transcribeAudio(transport, request.id, request.modelKey, request.audio, request.language);
		transport.send({ type: "transcription", id: request.id, text });
	} catch (error) {
		transport.send({ type: "error", id: request.id, error: errorText(error) });
	}
}

// ── Live streaming sessions ─────────────────────────────────────────

/** State for one in-flight native-online or endpointer-driven streaming session. */
interface StreamingSession {
	id: string;
	spec: SttModel;
	language: string | undefined;
	model: Promise<LoadedModel>;
	endpointer: StreamEndpointer;
	/** Online models consume each incoming chunk once, retaining one native stream. */
	audioQueue: Float32Array[];
	onlineStream?: SherpaOnlineStream;
	lastPartial: string;
	/** Finalized segments awaiting decode, in order. */
	segmentQueue: Float32Array[];
	/** Latest in-progress segment audio awaiting a volatile partial decode (coalesced). */
	pendingPartial: Float32Array | null;
	/** Committed segment transcripts, joined for the final result. */
	committed: string[];
	segmentIndex: number;
	pumping: boolean;
	cancelled: boolean;
	ended: boolean;
	/** Neural speech detector backing `speech_activity` events when enabled. */
	speechDetection: "energy" | "silero";
	/** Resolves once the detector is ready; `null` when disabled or load failed. */
	vad: Promise<SherpaVad | null>;
	/** Incoming audio queued for the detector until it finishes loading. */
	vadAudio: Float32Array[];
	/** Last reported detector state; emitted only on transitions. */
	vadActive: boolean;
	/** Set when Stop arrives while detector audio may still be queued. */
	vadFlush: boolean;
}

const sessions = new Map<string, StreamingSession>();

function startStreamingSession(
	transport: SttTransport,
	request: Extract<SttWorkerInbound, { type: "stream_start" }>,
): void {
	const spec = getSttModelSpec(request.modelKey);
	if (!spec) {
		transport.send({ type: "error", id: request.id, error: `Unknown stt model: ${request.modelKey}` });
		return;
	}
	const session: StreamingSession = {
		id: request.id,
		spec,
		language: request.language,
		// Defer loading until the session is registered; filter download callbacks too.
		model: Promise.resolve().then(() =>
			loadModel(
				request.modelKey,
				{
					...transport,
					send: message => {
						if (!session.cancelled) transport.send(message);
					},
				},
				request.id,
			),
		),
		endpointer: new StreamEndpointer(),
		audioQueue: [],
		lastPartial: "",
		segmentQueue: [],
		pendingPartial: null,
		committed: [],
		segmentIndex: 0,
		pumping: false,
		cancelled: false,
		ended: false,
		speechDetection: request.speechDetection === "silero" ? "silero" : "energy",
		vad: Promise.resolve(null),
		vadAudio: [],
		vadActive: false,
		vadFlush: false,
	};
	if (session.speechDetection === "silero") {
		const report = (detector: "silero" | "energy"): void => {
			if (sessions.get(request.id) === session && !session.cancelled) {
				transport.send({ type: "detector", id: request.id, detector });
			}
		};
		session.vad = createVad(request.modelKey, transport, request.id)
			.then(vad => {
				report("silero");
				return vad;
			})
			.catch(error => {
				// Detection degrades to energy rather than failing the session.
				sendLog(transport, "warn", "stt: neural speech detector unavailable", {
					error: errorMessage(error),
				});
				report("energy");
				return null;
			});
	}
	sessions.set(request.id, session);
	// Observe load failures even if no audio arrives (or startup is cancelled).
	void pumpSession(session, transport);
}

/**
 * Feed queued audio to the neural detector and report speech-state transitions.
 * Detector internals stay on the worker thread; only `active` edges cross IPC.
 */
async function drainVad(session: StreamingSession, transport: SttTransport): Promise<void> {
	if (session.speechDetection !== "silero") {
		session.vadAudio.length = 0;
		return;
	}
	const vad = await session.vad;
	if (!vad) {
		// Detector failed to load: stop queuing audio for it.
		session.speechDetection = "energy";
		session.vadAudio.length = 0;
		return;
	}
	if (session.vadAudio.length === 0 && !session.vadFlush) return;
	for (const audio of session.vadAudio) vad.acceptWaveform(audio);
	session.vadAudio.length = 0;
	// Honor a pending flush after feeding queued audio so an open speech run
	// closes promptly when the stream stops.
	if (session.vadFlush) {
		session.vadFlush = false;
		vad.flush();
	}
	// Drop finalized speech runs; only the open/closed state matters upstream.
	while (!vad.isEmpty()) vad.pop();
	const active = vad.isDetected();
	if (active !== session.vadActive) {
		session.vadActive = active;
		transport.send({ type: "speech_activity", id: session.id, active });
	}
}

/** Emit the closing speech-state edge when a session ends mid-utterance. */
function emitSpeechEnd(session: StreamingSession, transport: SttTransport): void {
	if (!session.vadActive) return;
	session.vadActive = false;
	transport.send({ type: "speech_activity", id: session.id, active: false });
}

function ingestStreamEvents(session: StreamingSession, events: EndpointerEvent[]): void {
	for (const event of events) {
		if (event.kind === "segment") session.segmentQueue.push(event.audio);
		else session.pendingPartial = event.audio;
	}
}

function commitOnlineSegment(session: StreamingSession, transport: SttTransport, text: string): void {
	if (text.length > 0) {
		session.committed.push(text);
		transport.send({ type: "segment", id: session.id, index: session.segmentIndex++, text });
	}
	session.lastPartial = "";
}

function decodeOnlineStream(
	session: StreamingSession,
	transport: SttTransport,
	recognizer: SherpaOnlineRecognizer,
	stream: SherpaOnlineStream,
	final: boolean,
): void {
	while (recognizer.isReady(stream)) {
		recognizer.decode(stream);
		const text = recognizer.getResult(stream).text.trim();
		if (!final && text !== session.lastPartial) {
			session.lastPartial = text;
			transport.send({ type: "partial", id: session.id, text });
		}
		if (!final && recognizer.isEndpoint(stream)) {
			commitOnlineSegment(session, transport, text);
			recognizer.reset(stream);
		}
	}
}

async function pumpOnlineSession(
	session: StreamingSession,
	transport: SttTransport,
	recognizer: SherpaOnlineRecognizer,
): Promise<void> {
	// Detector inference is cheap and independent of the ASR model lock.
	await drainVad(session, transport);
	await runOnModel(async () => {
		if (session.cancelled) return;
		const stream = (session.onlineStream ??= recognizer.createStream());
		for (const audio of session.audioQueue) {
			stream.acceptWaveform({ samples: audio, sampleRate: ASR_SAMPLE_RATE });
			decodeOnlineStream(session, transport, recognizer, stream, false);
		}
		session.audioQueue.length = 0;
		if (!session.ended) return;
		stream.acceptWaveform({ samples: ONLINE_TERMINAL_PADDING, sampleRate: ASR_SAMPLE_RATE });
		stream.inputFinished();
		decodeOnlineStream(session, transport, recognizer, stream, true);
		commitOnlineSegment(session, transport, recognizer.getResult(stream).text.trim());
		emitSpeechEnd(session, transport);
		transport.send({ type: "stream_done", id: session.id, text: session.committed.join(" ") });
		sessions.delete(session.id);
	});
}

/**
 * Drain a session's pending work: finalized segments first (committed in order),
 * then a single coalesced partial preview. Re-entrant-safe via `pumping`; new
 * audio that arrives mid-decode is picked up when the current decode resolves.
 */
async function pumpSession(session: StreamingSession, transport: SttTransport): Promise<void> {
	if (session.pumping) return;
	session.pumping = true;
	try {
		const model = await session.model;
		if (model.engine === "sherpa" && model.mode === "online") {
			// Audio or Stop can arrive while awaiting the model lock; drain again
			// until neither queued audio nor an unfinished Stop remains.
			while (!session.cancelled && sessions.get(session.id) === session) {
				await pumpOnlineSession(session, transport, model.recognizer);
				if (session.audioQueue.length === 0 && !session.ended) break;
			}
			// Detector work queued during the decode still needs its edges.
			await drainVad(session, transport);
			return;
		}
		while (!session.cancelled) {
			await drainVad(session, transport);
			if (session.segmentQueue.length > 0) {
				const audio = session.segmentQueue.shift()!;
				// A fresh segment supersedes any queued preview for the prior one.
				session.pendingPartial = null;
				const text = await runOnModel(() => decodeSegment(model, session.spec, audio, session.language));
				if (session.cancelled) return;
				if (text.length > 0) {
					session.committed.push(text);
					transport.send({ type: "segment", id: session.id, index: session.segmentIndex++, text });
				}
				continue;
			}
			if (session.pendingPartial) {
				const audio = session.pendingPartial;
				session.pendingPartial = null;
				const startedAt = performance.now();
				const text = await runOnModel(() => decodeSegment(model, session.spec, audio, session.language));
				if (session.cancelled) return;
				session.endpointer.notePartialDecode(performance.now() - startedAt);
				if (!session.endpointer.partialsEnabled) session.pendingPartial = null;
				// Skip a now-stale preview if a segment finalized mid-decode.
				if (text.length > 0 && session.segmentQueue.length === 0) {
					transport.send({ type: "partial", id: session.id, text });
				}
				continue;
			}
			break;
		}
		if (session.ended && !session.cancelled && session.segmentQueue.length === 0 && !session.pendingPartial) {
			emitSpeechEnd(session, transport);
			transport.send({ type: "stream_done", id: session.id, text: session.committed.join(" ") });
			sessions.delete(session.id);
		}
	} catch (error) {
		if (!session.cancelled) transport.send({ type: "error", id: session.id, error: errorText(error) });
		emitSpeechEnd(session, transport);
		sessions.delete(session.id);
	} finally {
		session.pumping = false;
	}
}

function handleStreamMessage(
	transport: SttTransport,
	message: Extract<
		SttWorkerInbound,
		{ type: "stream_start" | "stream_audio" | "stream_stop" | "stream_cancel" | "stream_detect" }
	>,
): void {
	if (message.type === "stream_start") {
		startStreamingSession(transport, message);
		return;
	}
	const session = sessions.get(message.id);
	if (!session || session.cancelled) return;
	switch (message.type) {
		case "stream_audio":
			if (session.ended) return;
			if (session.speechDetection === "silero") session.vadAudio.push(message.audio);
			if (session.spec.engine === "sherpa" && session.spec.mode === "online") {
				session.audioQueue.push(message.audio);
			} else {
				ingestStreamEvents(session, session.endpointer.push(message.audio));
			}
			void pumpSession(session, transport);
			return;
		case "stream_detect":
			if (session.ended) return;
			if (session.speechDetection === "silero") {
				session.vadAudio.push(message.audio);
				void pumpSession(session, transport);
			}
			return;
		case "stream_stop":
			if (session.ended) return;
			session.ended = true;
			session.pendingPartial = null;
			session.vadFlush = true;
			if (!(session.spec.engine === "sherpa" && session.spec.mode === "online")) {
				ingestStreamEvents(session, session.endpointer.flush());
			}
			void pumpSession(session, transport);
			return;
		case "stream_cancel":
			session.cancelled = true;
			session.audioQueue.length = 0;
			session.segmentQueue.length = 0;
			session.pendingPartial = null;
			session.vadAudio.length = 0;
			session.vadFlush = false;
			emitSpeechEnd(session, transport);
			sessions.delete(message.id);
			return;
	}
}

export function startSttWorker(transport: SttTransport): void {
	transport.onMessage(message => {
		switch (message.type) {
			case "ping":
				transport.send({ type: "pong", id: message.id });
				return;
			case "transcribe":
			case "download":
			case "vad_prepare":
				void handleBatchRequest(transport, message);
				return;
			default:
				handleStreamMessage(transport, message);
				return;
		}
	});
}

/** Process entry of `__omp_worker_stt`: serve the daemon `speech` module over stdio frames. */
export function runSttWorker(): Promise<void> {
	return runFramedStdioWorker<SttWorkerInbound, SttWorkerOutbound>(startSttWorker);
}
