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
	type: "string",
	default: DEFAULT_LIVE_VOICE,
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Voice",
		description: "Voice of realtime voice sessions; a custom live endpoint may take other names (set in config)",
		options: LIVE_VOICE_OPTIONS,
	},
});

function validateLiveEndpoint(raw: unknown): void {
	if (typeof raw !== "string" || !raw.trim()) return;
	let protocol: string;
	try {
		protocol = new URL(raw.trim()).protocol;
	} catch {
		throw new Error(`live.endpoint must be an http(s) URL, got ${JSON.stringify(raw)}`);
	}
	if (protocol !== "http:" && protocol !== "https:") {
		throw new Error(`live.endpoint must be an http(s) URL, got ${JSON.stringify(raw)}`);
	}
}

export const cfgLiveEndpoint = register({
	id: "live.endpoint",
	type: "string",
	default: undefined,
	validate: validateLiveEndpoint,
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Endpoint",
		description:
			"Base URL of an OpenAI Realtime WebRTC-compatible server (e.g. http://127.0.0.1:8080/v1) used for live voice instead of Codex; empty uses Codex",
	},
});

export const cfgLiveApiKey = register({
	id: "live.apiKey",
	type: "string",
	default: undefined,
	credential: true,
	ui: {
		tab: "providers",
		group: "Services",
		label: "Live Endpoint API Key",
		description: "Bearer key sent only to the live endpoint: a literal, an environment variable name, or !command",
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
			"Voice backend for /live: 'codex' uses OpenAI realtime; 'local' uses on-device speech recognition " +
			"and Kokoro playback with the session's configured model. Cloud models still receive the transcript.",
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

/** Absolute RMS floor for acoustic admission, before speech recognition. */
export const cfgLiveMicThreshold = register({
	id: "live.micThreshold",
	type: "number",
	default: 0.012,
	ui: {
		tab: "interaction",
		group: "Speech",
		label: "Live Microphone Threshold",
		description:
			"Local voice only. Higher values reject more background noise but can miss quiet speech. " +
			"This is an energy gate, not echo cancellation or a speech classifier. Takes effect next call.",
		options: [
			{ value: "0.008", label: "Quiet voice / close microphone (0.008)" },
			{ value: "0.012", label: "Normal (0.012)" },
			{ value: "0.025", label: "Noisy room (0.025)" },
		],
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
	return voice?.trim() || cfgLiveVoice.get(scope).trim() || DEFAULT_LIVE_VOICE;
}
