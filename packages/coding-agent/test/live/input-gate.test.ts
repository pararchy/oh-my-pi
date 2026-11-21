import { describe, expect, it } from "bun:test";
import { LiveInputGate } from "@oh-my-pi/pi-coding-agent/live/input-gate";

const FRAME = 320;
function frames(count: number, value: number) {
	return new Float32Array(count * FRAME).fill(value);
}
function collect() {
	const chunks: Float32Array[] = [];
	return {
		chunks,
		emit: (pcm: Float32Array) => chunks.push(pcm.slice()),
		values: () => chunks.flatMap(pcm => Array.from(pcm)),
	};
}

describe("local voice acoustic admission", () => {
	it("rejects quiet noise and isolated clicks instead of accumulating them into speech", () => {
		const gate = new LiveInputGate();
		const out = collect();
		for (let n = 0; n < 20; n++) {
			gate.push(frames(1, 0.4), out.emit);
			gate.push(frames(10, 0.003), out.emit);
		}
		expect(gate.active).toBe(false);
		expect(out.values().every(value => value === 0)).toBe(true);
	});
	it("requires sustained onset and preserves every first-phoneme sample exactly once", () => {
		const gate = new LiveInputGate({ preRollMs: 0 });
		const out = collect();
		const onset = new Float32Array(9 * FRAME);
		for (let n = 0; n < onset.length; n++) onset[n] = 0.1 + n / 100000;
		gate.push(onset.subarray(0, 8 * FRAME), out.emit);
		expect(out.values().every(value => value === 0)).toBe(true);
		gate.push(onset.subarray(8 * FRAME), out.emit);
		expect(gate.active).toBe(true);
		expect(out.values().filter(value => value !== 0)).toEqual(Array.from(onset));
	});
	it("replays bounded pre-roll before the admitted speech in chronological order", () => {
		const gate = new LiveInputGate({ preRollMs: 40, minSpeechMs: 40 });
		const out = collect();
		gate.push(frames(3, 0.001), out.emit);
		gate.push(frames(2, 0.2), out.emit);
		const admitted = out.values().filter(value => value !== 0);
		expect(admitted).toEqual([...frames(2, 0.001), ...frames(2, 0.2)]);
	});
	it("continues across a short pause and requires a new onset after release", () => {
		const gate = new LiveInputGate({ minSpeechMs: 40, releaseMs: 100, preRollMs: 0 });
		const out = collect();
		gate.push(frames(2, 0.2), out.emit);
		gate.push(frames(3, 0), out.emit);
		expect(gate.active).toBe(true);
		gate.push(frames(1, 0.3), out.emit);
		expect(gate.active).toBe(true);
		gate.push(frames(5, 0), out.emit);
		expect(gate.active).toBe(false);
		out.chunks.length = 0;
		gate.push(frames(1, 0.4), out.emit);
		expect(out.values().every(value => value === 0)).toBe(true);
		gate.push(frames(1, 0.5), out.emit);
		expect(out.values().filter(value => value !== 0)).toEqual([...frames(1, 0.4), ...frames(1, 0.5)]);
	});
	it("has identical PCM behavior with split frames and non-zero input byte offsets", () => {
		const pcm = new Float32Array(30 * FRAME);
		pcm.set(frames(15, 0.2), 3 * FRAME);
		const whole = collect();
		const a = new LiveInputGate();
		a.push(pcm, whole.emit);
		const split = collect();
		const b = new LiveInputGate();
		for (let n = 0; n < pcm.length; n += 137) b.push(pcm.subarray(n, n + 137), split.emit);
		expect(split.values()).toEqual(whole.values());
	});
	it("reset discards both partial frames and a pending onset", () => {
		const gate = new LiveInputGate({ minSpeechMs: 40 });
		const out = collect();
		gate.push(frames(1, 0.6), out.emit);
		gate.push(new Float32Array(113).fill(0.6), out.emit);
		gate.reset();
		out.chunks.length = 0;
		gate.push(frames(2, 0.2), out.emit);
		expect(out.values().filter(value => value !== 0)).toEqual(Array.from(frames(2, 0.2)));
	});
	it("owns onset PCM even if a capture buffer is immediately reused", () => {
		const gate = new LiveInputGate({ minSpeechMs: 40, preRollMs: 0 });
		const out = collect();
		const capture = frames(1, 0.2);
		gate.push(capture, out.emit);
		capture.fill(0.4);
		gate.push(capture, out.emit);
		expect(out.values().filter(value => value !== 0)).toEqual([...frames(1, 0.2), ...frames(1, 0.4)]);
	});
});
