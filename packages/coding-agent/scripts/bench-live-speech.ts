/**
 * Compare local speech-to-text tiers on identical WAV input.
 *
 * Feeds 16 kHz mono PCM through the same streaming session path the live voice
 * surface uses (`SttClient.startStream`), then reports per-file wall time,
 * real-time factor, committed segments, and the final transcript. When a
 * `<file>.txt` sibling holds a reference transcript, a normalized word-error
 * rate is included.
 *
 * Usage:
 *   bun scripts/bench-live-speech.ts [--models key,...] [--repeats N] file.wav...
 *   bun scripts/bench-live-speech.ts --vad file.wav   # detector edges only
 */
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { SttClient, type SttStreamHandle } from "../src/stt/asr-client";
import { STT_MODEL_VALUES, type SttModelKey } from "../src/stt/models";

const SAMPLE_RATE = 16_000;
/** Chunk size pushed per tick; mirrors the recorder cadence. */
const CHUNK_MS = 40;

interface SpeechEdge {
	atMs: number;
	active: boolean;
}

interface RunResult {
	model: SttModelKey | "vad";
	file: string;
	wallMs: number;
	rtf: number;
	segments: string[];
	text: string;
	detector: string;
	speechEdges: SpeechEdge[];
	wer?: number;
}

// ── WAV decoding ──────────────────────────────────────────────────────

/** Minimal RIFF/WAVE reader: PCM 8/16/24/32-bit int and 32-bit float, any rate/channels. */
function decodeWav(buffer: Buffer): { samples: Float32Array; sampleRate: number } {
	if (buffer.length < 44 || buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
		throw new Error("not a RIFF/WAVE file");
	}
	let offset = 12;
	let format = 0;
	let channels = 0;
	let sampleRate = 0;
	let bits = 0;
	let dataStart = 0;
	let dataLen = 0;
	while (offset + 8 <= buffer.length) {
		const id = buffer.toString("ascii", offset, offset + 4);
		const size = buffer.readUInt32LE(offset + 4);
		const body = offset + 8;
		if (id === "fmt ") {
			format = buffer.readUInt16LE(body);
			channels = buffer.readUInt16LE(body + 2);
			sampleRate = buffer.readUInt32LE(body + 4);
			bits = buffer.readUInt16LE(body + 14);
		} else if (id === "data") {
			dataStart = body;
			dataLen = size;
			break;
		}
		offset = body + size + (size % 2);
	}
	if (!dataStart || dataLen <= 0) throw new Error("no data chunk");
	const bytesPerSample = bits / 8;
	const frames = Math.floor(dataLen / (bytesPerSample * channels));
	const out = new Float32Array(frames);
	for (let frame = 0; frame < frames; frame += 1) {
		let sum = 0;
		for (let ch = 0; ch < channels; ch += 1) {
			const at = dataStart + (frame * channels + ch) * bytesPerSample;
			if (format === 3 && bits === 32) sum += buffer.readFloatLE(at);
			else if (bits === 16) sum += buffer.readInt16LE(at) / 0x8000;
			else if (bits === 8) sum += (buffer.readUInt8(at) - 128) / 128;
			else if (bits === 24) {
				const v = buffer.readIntLE(at, 3);
				sum += v / 0x80_0000;
			} else if (bits === 32) sum += buffer.readInt32LE(at) / 0x80_0000;
		}
		out[frame] = sum / channels;
	}
	return { samples: out, sampleRate };
}

/** Linear-interpolation resample to 16 kHz (adequate for speech benchmarks). */
function resample16k(samples: Float32Array, rate: number): Float32Array {
	if (rate === SAMPLE_RATE) return samples;
	const frames = Math.max(1, Math.round((samples.length * SAMPLE_RATE) / rate));
	const out = new Float32Array(frames);
	const step = rate / SAMPLE_RATE;
	for (let index = 0; index < frames; index += 1) {
		const at = index * step;
		const floor = Math.floor(at);
		const frac = at - floor;
		const a = samples[floor] ?? 0;
		const b = samples[floor + 1] ?? a;
		out[index] = a + (b - a) * frac;
	}
	return out;
}

// ── Word error rate ───────────────────────────────────────────────────

function words(text: string): string[] {
	return text
		.toUpperCase()
		.replace(/[^A-Z0-9' ]+/g, " ")
		.split(/\s+/)
		.filter(Boolean);
}

/** Levenshtein over word arrays; returns errors / reference words. */
function wordErrorRate(reference: string, hypothesis: string): number {
	const ref = words(reference);
	const hyp = words(hypothesis);
	if (ref.length === 0) return 0;
	let prev = Array.from({ length: hyp.length + 1 }, (_unused, j) => j);
	for (let i = 1; i <= ref.length; i += 1) {
		const cur = Array.from<number>({ length: hyp.length + 1 });
		cur[0] = i;
		for (let j = 1; j <= hyp.length; j += 1) {
			cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
		}
		prev = cur;
	}
	return prev[hyp.length]! / ref.length;
}

// ── Runner ────────────────────────────────────────────────────────────

async function runStream(
	client: SttClient,
	model: SttModelKey,
	pcm: Float32Array,
	language: string | undefined,
): Promise<Omit<RunResult, "model" | "file" | "rtf" | "wer">> {
	const segments: string[] = [];
	const speechEdges: SpeechEdge[] = [];
	let fedSamples = 0;
	let detector = "none";
	const stream: SttStreamHandle = client.startStream(model, {
		language,
		speechDetection: "silero",
		onDetector: resolved => {
			detector = resolved;
		},
		onSpeech: active => {
			speechEdges.push({ atMs: Math.round((fedSamples / SAMPLE_RATE) * 1000), active });
		},
		onSegment: text => segments.push(text),
	});
	const startedAt = performance.now();
	const chunkSamples = Math.round((SAMPLE_RATE * CHUNK_MS) / 1000);
	for (let offset = 0; offset < pcm.length; offset += chunkSamples) {
		const chunk = pcm.subarray(offset, offset + chunkSamples);
		fedSamples += chunk.length;
		stream.pushDetect(chunk);
		stream.pushAudio(chunk);
	}
	const text = await stream.stop();
	return {
		wallMs: performance.now() - startedAt,
		segments,
		text,
		detector,
		speechEdges,
	};
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	let models: SttModelKey[] = STT_MODEL_VALUES.filter(key => !key.startsWith("nemotron"));
	let repeats = 1;
	let vadOnly = false;
	let language: string | undefined = "en";
	const files: string[] = [];
	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i]!;
		if (arg === "--models") models = args[++i]!.split(",") as SttModelKey[];
		else if (arg === "--repeats") repeats = Number(args[++i]);
		else if (arg === "--language") language = args[++i]!;
		else if (arg === "--vad") vadOnly = true;
		else files.push(arg);
	}
	if (files.length === 0) {
		console.error("usage: bench-live-speech.ts [--models k,...] [--repeats N] [--vad] file.wav...");
		process.exit(2);
	}

	const client = new SttClient();
	const results: RunResult[] = [];
	try {
		for (const file of files) {
			const wav = decodeWav(await fs.readFile(file));
			const pcm = resample16k(wav.samples, wav.sampleRate);
			const seconds = pcm.length / SAMPLE_RATE;
			const refPath = `${file}.txt`;
			const reference = await fs.readFile(refPath, "utf8").catch(() => undefined);
			for (const model of vadOnly ? (["vad"] as const) : models) {
				for (let run = 0; run < repeats; run += 1) {
					const result = vadOnly
						? await runVadOnly(client, pcm)
						: await runStream(client, model as SttModelKey, pcm, language);
					const entry: RunResult = {
						model: vadOnly ? "vad" : (model as SttModelKey),
						file: path.basename(file),
						wallMs: result.wallMs,
						rtf: result.wallMs / (seconds * 1000),
						segments: result.segments,
						text: result.text,
						detector: result.detector,
						speechEdges: result.speechEdges,
						...(reference !== undefined ? { wer: wordErrorRate(reference, result.text) } : {}),
					};
					results.push(entry);
					console.log(
						`${entry.model} ${entry.file}: ${seconds.toFixed(1)}s audio, ${entry.wallMs.toFixed(0)}ms ` +
							`(rtf ${entry.rtf.toFixed(2)}), detector=${entry.detector}, ` +
							`edges=${entry.speechEdges.length}` +
							(entry.wer !== undefined ? `, wer=${(entry.wer * 100).toFixed(1)}%` : ""),
					);
					console.log(`  text: ${entry.text || "(empty)"}`);
				}
			}
		}
	} finally {
		await client.terminate();
	}
}

/** Detector-only pass: audio goes only through `pushDetect`, no recognition. */
async function runVadOnly(
	client: SttClient,
	pcm: Float32Array,
): Promise<Omit<RunResult, "model" | "file" | "rtf" | "wer">> {
	const speechEdges: SpeechEdge[] = [];
	let fedSamples = 0;
	let detector = "none";
	const stream = client.startStream("whisper-base", {
		speechDetection: "silero",
		onDetector: resolved => {
			detector = resolved;
		},
		onSpeech: active => {
			speechEdges.push({ atMs: Math.round((fedSamples / SAMPLE_RATE) * 1000), active });
		},
	});
	const startedAt = performance.now();
	const chunkSamples = Math.round((SAMPLE_RATE * CHUNK_MS) / 1000);
	for (let offset = 0; offset < pcm.length; offset += chunkSamples) {
		const chunk = pcm.subarray(offset, offset + chunkSamples);
		fedSamples += chunk.length;
		stream.pushDetect(chunk);
	}
	// Flush the detector via stop; the recognizer still closes cleanly.
	const text = await stream.stop();
	return { wallMs: performance.now() - startedAt, segments: [], text, detector, speechEdges };
}

await main();
