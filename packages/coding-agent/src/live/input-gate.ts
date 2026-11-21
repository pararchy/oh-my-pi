/** Amplitude/duration admission before ASR, not a neural speech detector or AEC. */
export interface LiveInputGateOptions {
	threshold?: number;
	minSpeechMs?: number;
	releaseMs?: number;
	preRollMs?: number;
	sampleRate?: number;
}

/**
 * Analyze fixed 20ms frames, preserving onset and short pauses in bounded PCM.
 * Closed frames emit silence. Once sustained energy admits an utterance, replay
 * its owned pre-roll exactly once, then pass audio until the release expires.
 * `emit` must consume/copy synchronously: internal frame/ring views are reused.
 */
export class LiveInputGate {
	readonly #threshold: number;
	readonly #frame: Float32Array;
	readonly #silence: Float32Array;
	readonly #preRoll: Float32Array;
	readonly #onsetFrames: number;
	readonly #releaseFrames: number;
	#frameLength = 0;
	#write = 0;
	#retained = 0;
	#onset = 0;
	#quiet = 0;
	#active = false;

	constructor(options: LiveInputGateOptions = {}) {
		const sampleRate = options.sampleRate ?? 16000;
		const threshold = options.threshold ?? 0.012;
		const minSpeechMs = options.minSpeechMs ?? 180;
		const releaseMs = options.releaseMs ?? 600;
		const preRollMs = options.preRollMs ?? 240;
		if (
			!Number.isFinite(sampleRate) ||
			sampleRate < 1000 ||
			sampleRate > 192000 ||
			!Number.isFinite(threshold) ||
			threshold <= 0 ||
			threshold > 1 ||
			!Number.isFinite(minSpeechMs) ||
			minSpeechMs <= 0 ||
			minSpeechMs > 2000 ||
			!Number.isFinite(releaseMs) ||
			releaseMs < 0 ||
			releaseMs > 5000 ||
			!Number.isFinite(preRollMs) ||
			preRollMs < 0 ||
			preRollMs > 2000
		) {
			throw new RangeError("Invalid local live microphone gate configuration");
		}
		const frameSamples = Math.round(sampleRate * 0.02);
		this.#threshold = threshold;
		this.#frame = new Float32Array(frameSamples);
		this.#silence = new Float32Array(frameSamples);
		this.#onsetFrames = Math.ceil(minSpeechMs / 20);
		this.#releaseFrames = Math.ceil(releaseMs / 20);
		// Retain the pending onset even when preRollMs is shorter than minSpeechMs.
		this.#preRoll = new Float32Array((Math.ceil(preRollMs / 20) + this.#onsetFrames) * frameSamples);
	}

	get active(): boolean {
		return this.#active;
	}

	reset(): void {
		this.#frameLength = 0;
		this.#write = 0;
		this.#retained = 0;
		this.#onset = 0;
		this.#quiet = 0;
		this.#active = false;
	}

	push(samples: Float32Array, emit: (samples: Float32Array) => void): void {
		let offset = 0;
		while (offset < samples.length) {
			const length = Math.min(this.#frame.length - this.#frameLength, samples.length - offset);
			this.#frame.set(samples.subarray(offset, offset + length), this.#frameLength);
			this.#frameLength += length;
			offset += length;
			if (this.#frameLength < this.#frame.length) continue;
			this.#frameLength = 0;
			this.#processFrame(emit);
		}
	}

	#processFrame(emit: (samples: Float32Array) => void): void {
		let sumSquares = 0;
		for (const value of this.#frame) sumSquares += value * value;
		const voiced = Math.sqrt(sumSquares / this.#frame.length) >= this.#threshold;
		if (this.#active) {
			this.#quiet = voiced ? 0 : this.#quiet + 1;
			emit(this.#frame);
			if (!voiced && this.#quiet >= this.#releaseFrames) {
				this.#active = false;
				this.#quiet = 0;
				this.#onset = 0;
				this.#retained = 0;
				this.#write = 0;
			}
			return;
		}

		this.#preRoll.set(this.#frame, this.#write);
		this.#write = (this.#write + this.#frame.length) % this.#preRoll.length;
		this.#retained = Math.min(this.#preRoll.length, this.#retained + this.#frame.length);
		this.#onset = voiced ? this.#onset + 1 : 0;
		if (this.#onset < this.#onsetFrames) {
			emit(this.#silence);
			return;
		}

		this.#active = true;
		this.#quiet = 0;
		const start = (this.#write - this.#retained + this.#preRoll.length) % this.#preRoll.length;
		const firstLength = Math.min(this.#retained, this.#preRoll.length - start);
		emit(this.#preRoll.subarray(start, start + firstLength));
		if (firstLength < this.#retained) emit(this.#preRoll.subarray(0, this.#retained - firstLength));
		this.#retained = 0;
	}
}
