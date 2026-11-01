import { beforeAll, describe, expect, it } from "bun:test";
import { Container } from "../src/tui";
import { Text as PlainText } from "../src/components/text";
import { TranscriptContainer } from "../src/chrome/transcript-container";
import { GrimoireLayout } from "../src/prompt/grimoire-layout";
import { initTheme } from "../src/theme/theme";

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
});
