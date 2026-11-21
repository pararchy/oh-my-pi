/**
 * Process-wide "a live call is active" signal. The visualizer's own toast-free
 * UX makes per-turn completion/error desktop notifications redundant — and
 * harmful: stray speech turns during a call would spam one toast per turn.
 */
let activeLiveCalls = 0;

export function enterLiveCall(): () => void {
	activeLiveCalls += 1;
	let released = false;
	return () => {
		if (released) return;
		released = true;
		activeLiveCalls = Math.max(0, activeLiveCalls - 1);
	};
}

export function liveCallActive(): boolean {
	return activeLiveCalls > 0;
}
