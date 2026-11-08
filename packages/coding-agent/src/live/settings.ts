/**
 * Settings declared by this domain (see `config/registry.ts`). Declaration order is the
 * settings-panel order; `config/all-settings.ts` registers every domain.
 */
import { register, type ScopeLike } from "../config/registry";
import { TTS_LOCAL_VOICE_OPTIONS, TTS_LOCAL_VOICE_VALUES } from "../tts/models";
import { DEFAULT_LIVE_VOICE, LIVE_VOICE_OPTIONS, LIVE_VOICE_VALUES } from "./voices";

export type LiveEngine = "codex" | "local";

export const cfgLiveVoice = register({
	id: "live.voice",
	type: "enum",
	values: LIVE_VOICE_VALUES,
	default: DEFAULT_LIVE_VOICE,
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Voice",
		description: "Voice used by Codex-backed realtime voice sessions",
		options: LIVE_VOICE_OPTIONS,
	},
});

/** Realtime backend for `/live`: the remote Codex surface or the local STT→session→TTS loop. */
export const cfgLiveEngine = register({
	id: "live.engine",
	type: "enum",
	values: ["codex", "local"],
	default: "codex",
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Engine",
		description:
			"Voice backend for /live: 'codex' uses the remote realtime model; 'local' runs fully on-device " +
			"(local speech-to-text, the session's own model, local Kokoro voice) so persona and tools carry over",
		options: [
			{ value: "codex", label: "Codex (remote realtime model)" },
			{ value: "local", label: "Local (on-device STT + session model + Kokoro)" },
		],
	},
});

/** Kokoro voice for the local engine; empty falls back to `speech.voice`, then the model default. */
export const cfgLiveLocalVoice = register({
	id: "live.localVoice",
	type: "enum",
	values: ["", ...TTS_LOCAL_VOICE_VALUES],
	default: "",
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Local Voice",
		description: "Kokoro voice for the local /live engine; empty uses the Speech Vocalization Voice setting",
		options: [{ value: "", label: "Same as Speech Vocalization Voice" }, ...TTS_LOCAL_VOICE_OPTIONS],
	},
});

/** Persona-specific speech style; supplements rather than replaces the live protocol prompt. */
export const cfgLivePersonality = register({
	id: "live.personality",
	type: "string",
	default: "",
});

/** The voice a live session uses: an explicit choice, else the `live.voice` setting, else the default. */
export function resolveLiveVoice(scope: ScopeLike, voice?: string): string {
	return voice?.trim() || cfgLiveVoice.get(scope) || DEFAULT_LIVE_VOICE;
}
