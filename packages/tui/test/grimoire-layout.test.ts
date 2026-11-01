import { beforeAll, describe, expect, it } from "bun:test";
import { stripVTControlCharacters as stripAnsi } from "node:util";
import { Container } from "../src/tui";
import { Text as PlainText } from "../src/components/text";
import { TranscriptContainer } from "../src/chrome/transcript-container";
import { GrimoireLayout } from "../src/prompt/grimoire-layout";
import { initTheme } from "../src/theme/theme";
import { visibleWidth } from "../src/utils";

beforeAll(() => initTheme());

describe("Grimoire fixed-top viewport", () => {
	it("scrolls the conversation without moving live chrome or duplicating the draft", () => {
		const layout = new GrimoireLayout();
		const transcript = new TranscriptContainer();
		for (let i = 1; i <= 50; i++) transcript.addChild(new PlainText(`row-${i}`, 0, 0));
		const editor = new PlainText("unfinished draft", 0, 0);
		const prompt = new Container();
		prompt.addChild(editor);
		const roots = [new PlainText("BANNER", 0, 0), transcript, new PlainText("STATUS", 0, 0), prompt];
		const initial = layout.render(roots, editor, 80, 15).map(row => row.trimEnd());
		expect(initial.slice(0, 2)).toEqual(["BANNER", "STATUS"]);
		expect(initial.at(-1)).toBe("unfinished draft");
		expect(initial.join("\n")).toContain("row-50");
		layout.scroll(10);
		const scrolled = layout.render(roots, editor, 80, 15).map(row => row.trimEnd());
		expect(scrolled.slice(0, 2)).toEqual(["BANNER", "STATUS"]);
		expect(scrolled.at(-1)).toBe("unfinished draft");
		expect(scrolled.join("\n")).not.toContain("row-50");
		expect(scrolled.filter(row => row === "unfinished draft")).toEqual(["unfinished draft"]);
	});

	it("keeps a question above the conversation and the draft separate when resized", () => {
		const layout = new GrimoireLayout();
		const transcript = new TranscriptContainer();
		transcript.addChild(new PlainText("conversation", 0, 0));
		const editor = new PlainText("draft", 0, 0);
		const prompt = new Container();
		prompt.addChild(new PlainText("QUESTION", 0, 0));
		prompt.addChild(editor);
		const roots = [new PlainText("FACE", 0, 0), transcript, prompt];
		for (const rows of [20, 8, 5]) {
			const frame = layout.render(roots, editor, 60, rows).map(row => row.trimEnd());
			expect(frame[0]).toBe("FACE");
			expect(frame[1]).toBe("QUESTION");
			expect(frame.at(-1)).toBe("draft");
			expect(frame.length).toBe(rows);
		}
	});
	it("keeps the spirit centered between facts while dictation is centered and attachments stay left", () => {
		const layout = new GrimoireLayout();
		const banner = {
			render: (width: number) => [" ".repeat(Math.floor((width - 2) / 2)) + "\x1b[32m界\x1b[0m"],
		};
		const status = {
			render: () => ["old status must not be duplicated"],
			renderGrimoire: () => ({ left: ["model", "directory"], right: ["context"], below: ["mic recording"] }),
		};
		const editor = new PlainText("draft", 0, 0);
		const image = new PlainText("IMAGE #1", 0, 0);
		const roots = [banner, new TranscriptContainer(), image, editor, status];
		const frame = layout.render(roots, editor, 120, 30, banner, status);
		const plain = frame.map(stripAnsi);
		const faceRow = plain.find(row => row.includes("界"))!;
		expect(faceRow.indexOf("界")).toBe(59);
		expect(faceRow.indexOf("model")).toBeLessThan(faceRow.indexOf("界"));
		expect(faceRow.indexOf("context")).toBeGreaterThan(faceRow.indexOf("界"));
		expect(plain.find(row => row.includes("mic recording"))?.indexOf("mic recording")).toBe(53);
		expect(plain.find(row => row.includes("IMAGE #1"))?.indexOf("IMAGE #1")).toBe(0);
		expect(plain.join("\n")).not.toContain("old status");
		expect(frame.every(row => visibleWidth(row) <= 120)).toBe(true);
		expect(frame.length).toBe(30);
	});

	it("stacks facts on narrow terminals without losing the dialog or editing the parked draft", () => {
		const layout = new GrimoireLayout();
		const banner = new PlainText("face", 0, 0);
		const status = {
			render: () => [],
			renderGrimoire: () => ({ left: ["model", "directory"], right: ["context", "MCP offline"], below: ["mic"] }),
		};
		const editor = new PlainText("parked draft", 0, 0);
		const prompt = new Container();
		prompt.addChild(new PlainText("QUESTION", 0, 0));
		prompt.addChild(editor);
		const roots = [banner, new TranscriptContainer(), prompt, status];
		for (const width of [40, 80, 120]) {
			const frame = layout.render(roots, editor, width, 20, banner, status);
			const plain = frame.map(stripAnsi);
			expect(plain.join("\n")).toContain("directory");
			expect(plain.join("\n")).toContain("MCP offline");
			expect(plain.filter(row => row.includes("QUESTION"))).toHaveLength(1);
			expect(plain.at(-1)?.trimEnd()).toBe("parked draft");
			expect(frame.every(row => visibleWidth(row) <= width)).toBe(true);
			expect(frame.length).toBe(20);
		}
	});
});
