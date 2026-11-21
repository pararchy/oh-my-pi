import { beforeAll, describe, expect, it } from "bun:test";
import { stripVTControlCharacters as stripAnsi } from "node:util";
import { Container } from "../src/tui";
import { Text as PlainText } from "../src/components/text";
import { TranscriptContainer } from "../src/chrome/transcript-container";
import { GrimoireLayout } from "../src/prompt/grimoire-layout";
import { initTheme } from "../src/theme/theme";
import { visibleWidth } from "../src/utils";
import { Composer } from "../src/prompt/composer";
import { VirtualTerminal } from "./virtual-terminal";

class WheelTerminal extends VirtualTerminal {
	#reporting = false;
	override write(data: string): void {
		for (const match of data.matchAll(/\x1b\[\?1000([hl])/g)) this.#reporting = match[1] === "h";
		super.write(data);
	}
	wheel(button: 64 | 65): void {
		// A terminal sends wheel reports only while the application requests them.
		if (this.#reporting) this.sendInput(`\x1b[<${button};20;10M`);
	}
}

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

	it("receives wheel gestures before and after a picker without moving the header or editing the draft", () => {
		const previousLayout = process.env.PARA_OMP_LAYOUT;
		process.env.PARA_OMP_LAYOUT = "top";
		const terminal = new WheelTerminal(80, 24);
		const composer = new Composer({ terminal, preferences: { quiet: true } });
		if (previousLayout === undefined) delete process.env.PARA_OMP_LAYOUT;
		else process.env.PARA_OMP_LAYOUT = previousLayout;
		const transcript = new TranscriptContainer();
		for (let i = 1; i <= 80; i++) transcript.addChild(new PlainText(`message-${i}`, 0, 0));
		const header = new PlainText("FIXED HEADER", 0, 0);
		composer.editor.setText("unfinished draft");
		composer.setRuntimeChildren([header, transcript, composer.editor], { fixedHeader: header });
		composer.start({ playWelcomeIntro: false });
		const paint = () => {
			composer.ui.renderNow();
			return terminal.getViewport();
		};
		try {
			const tail = paint();
			expect(tail.join("\n")).toContain("message-80");
			terminal.wheel(64);
			const earlier = paint();
			expect(earlier.join("\n")).toContain("message-77");
			expect(earlier.join("\n")).not.toContain("message-80");
			expect(earlier[0]).toContain("FIXED HEADER");
			expect(earlier[0]).toBe(tail[0]);
			expect(composer.editor.getText()).toBe("unfinished draft");
			terminal.wheel(65);
			expect(paint()).toEqual(tail);

			const picker = composer.ui.showOverlay(new PlainText("PICKER", 0, 0), {
				fullscreen: true,
				mouseTracking: false,
			});
			paint();
			terminal.wheel(64);
			picker.hide();
			expect(paint()).toEqual(tail);
			terminal.wheel(64);
			expect(paint()).toEqual(earlier);
			expect(composer.editor.getText()).toBe("unfinished draft");
		} finally {
			composer.stop();
		}
	});
});
