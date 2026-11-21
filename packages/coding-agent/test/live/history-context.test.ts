import { describe, expect, it } from "bun:test";
import {
	buildLiveHistoryContext,
	LIVE_TRANSCRIPT_MESSAGE_TYPE,
	persistLiveTranscriptTurn,
} from "@oh-my-pi/pi-coding-agent/live/controller";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage } from "@oh-my-pi/pi-ai";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";

const extract = (message: AssistantMessage): string =>
	message.content
		.flatMap(block => (block.type === "text" ? [block.text] : []))
		.join("\n");

function user(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: 1 } as unknown as AgentMessage;
}
function assistant(text: string): AgentMessage {
	return { role: "assistant", content: [{ type: "text", text }], timestamp: 1 } as unknown as AgentMessage;
}
function voiceTurn(role: "user" | "assistant", text: string): AgentMessage {
	return {
		role: "custom",
		customType: LIVE_TRANSCRIPT_MESSAGE_TYPE,
		content: `[Voice ${role}] ${text}`,
		display: false,
		details: { role },
		attribution: "agent",
		timestamp: 1,
	} as unknown as AgentMessage;
}
function otherCustom(text: string): AgentMessage {
	return {
		role: "custom",
		customType: "live-delegation",
		content: text,
		display: false,
		timestamp: 1,
	} as unknown as AgentMessage;
}
function toolResult(): AgentMessage {
	return { role: "toolResult", toolCallId: "1", toolName: "read", content: [], isError: false, timestamp: 1 } as unknown as AgentMessage;
}

describe("buildLiveHistoryContext", () => {
	it("replays recent conversation oldest-to-newest with labels", () => {
		const context = buildLiveHistoryContext(
			[user("first question"), assistant("first answer"), user("second question")],
			extract,
		);
		expect(context).toBeDefined();
		const first = context!.indexOf("User: first question");
		const answer = context!.indexOf("Assistant: first answer");
		const last = context!.indexOf("User: second question");
		expect(first).toBeGreaterThanOrEqual(0);
		expect(first).toBeLessThan(answer);
		expect(answer).toBeLessThan(last);
	});

	it("labels persisted voice turns with their speaker", () => {
		const context = buildLiveHistoryContext(
			[voiceTurn("user", "said aloud"), voiceTurn("assistant", "replied aloud")],
			extract,
		);
		expect(context).toContain("User (voice): [Voice user] said aloud");
		expect(context).toContain("Assistant (voice): [Voice assistant] replied aloud");
	});

	it("excludes tool results and unrelated custom records", () => {
		const context = buildLiveHistoryContext(
			[otherCustom("delegated task"), toolResult(), user("keep me")],
			extract,
		);
		expect(context).toContain("User: keep me");
		expect(context).not.toContain("delegated task");
		expect(context).not.toContain("toolResult");
	});

	it("returns undefined for an empty session", () => {
		expect(buildLiveHistoryContext([], extract)).toBeUndefined();
		expect(buildLiveHistoryContext([toolResult()], extract)).toBeUndefined();
	});

	it("keeps the newest turns when the history is long", () => {
		const messages: AgentMessage[] = [];
		for (let i = 0; i < 30; i += 1) {
			messages.push(user(`question ${i}`), assistant(`answer ${i}`));
		}
		const context = buildLiveHistoryContext(messages, extract)!;
		expect(context).toContain("Assistant: answer 29");
		expect(context).not.toContain("question 0");
		expect(context).not.toContain("question 10");
	});
});

describe("persistLiveTranscriptTurn", () => {
	it("appends a hidden custom transcript entry", async () => {
		const sent: unknown[] = [];
		const session = {
			async sendCustomMessage(message: unknown) {
				sent.push(message);
				return false;
			},
		} as unknown as Pick<AgentSession, "sendCustomMessage">;
		persistLiveTranscriptTurn(session, "user", "what is the status");
		await Promise.resolve();
		expect(sent).toHaveLength(1);
		const payload = sent[0] as { customType: string; content: string; display: boolean; details: { role: string } };
		expect(payload.customType).toBe(LIVE_TRANSCRIPT_MESSAGE_TYPE);
		expect(payload.content).toBe("[Voice user] what is the status");
		expect(payload.display).toBe(false);
		expect(payload.details.role).toBe("user");
	});

	it("does not throw when the session rejects the append", async () => {
		const called = Promise.withResolvers<void>();
		const session = {
			async sendCustomMessage() {
				called.resolve();
				throw new Error("session closed");
			},
		} as unknown as Pick<AgentSession, "sendCustomMessage">;
		expect(() => persistLiveTranscriptTurn(session, "assistant", "done")).not.toThrow();
		await called.promise;
	});
});
