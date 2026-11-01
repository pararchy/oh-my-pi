import type { Component } from "../tui";
import { Container } from "../tui";
import { TranscriptContainer } from "../chrome/transcript-container";

/** Fullscreen layout: live chrome and dialogs never enter transcript history. */
export class GrimoireLayout {
	#offset = 0;
	#pageRows = 10;

	scroll(delta: number): void {
		this.#offset = Math.max(0, this.#offset + delta);
	}

	get pageRows(): number { return this.#pageRows; }

	render(roots: readonly Component[], editor: Component, width: number, height: number): string[] {
		if (height <= 0) return [];
		const transcript = roots.find(root => root instanceof TranscriptContainer) as TranscriptContainer | undefined;
		const top: string[] = [];
		const dialogs: string[] = [];
		// Dialogs replace children in the editor container, never the editor object.
		for (const root of roots) {
			if (root === transcript || root === editor) continue;
			if (root instanceof Container && root.children.some(child => child === editor || child.handleInput)) {
				for (const child of root.children) if (child !== editor) dialogs.push(...child.render(width));
			} else top.push(...root.render(width));
		}
		const prompt = editor.render(width).slice(-Math.max(1, Math.floor(height / 3)));
		const chromeBudget = Math.max(0, height - prompt.length - 1);
		// Questions take precedence on short terminals; trim decorative chrome first.
		const questionRows = dialogs.slice(0, chromeBudget);
		const header = top.slice(0, Math.max(0, chromeBudget - questionRows.length));
		const bodyRows = Math.max(0, height - header.length - questionRows.length - prompt.length);
		this.#pageRows = Math.max(1, bodyRows - 1);
		const tail = transcript?.renderTail(width, bodyRows + this.#offset) ?? [];
		this.#offset = Math.min(this.#offset, Math.max(0, tail.length - bodyRows));
		const end = tail.length - this.#offset;
		const body = tail.slice(Math.max(0, end - bodyRows), end);
		return [...header, ...questionRows, ...body, ...Array<string>(Math.max(0, bodyRows - body.length)).fill(""), ...prompt];
	}
}
