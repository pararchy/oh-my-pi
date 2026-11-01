import type { Component } from "../tui";
import { Container } from "../tui";
import { TranscriptContainer } from "../chrome/transcript-container";
import { truncateToWidth, visibleWidth, wrapTextWithAnsi } from "../utils";

export interface GrimoireStatus {
	renderGrimoire(width: number): { left: readonly string[]; right: readonly string[]; below: readonly string[] } | undefined;
}

const centered = (text: string, width: number): string => {
	const fitted = truncateToWidth(text, width);
	return " ".repeat(Math.max(0, Math.floor((width - visibleWidth(fitted)) / 2))) + fitted;
};

/** Fullscreen layout: live chrome and dialogs never enter transcript history. */
export class GrimoireLayout {
	#offset = 0;
	#pageRows = 10;

	scroll(delta: number): void {
		this.#offset = Math.max(0, this.#offset + delta);
	}

	get pageRows(): number { return this.#pageRows; }

	render(
		roots: readonly Component[], editor: Component, width: number, height: number,
		banner?: Component, status?: Component & GrimoireStatus,
	): string[] {
		if (height <= 0) return [];
		const transcript = roots.find(root => root instanceof TranscriptContainer) as TranscriptContainer | undefined;
		const top: string[] = [];
		const dialogs: string[] = [];
		const sideWidth = Math.floor((width - 44) / 2);
		const wide = sideWidth >= 24 && height >= 24;
		const columns = banner && status ? status.renderGrimoire(wide ? sideWidth : width) : undefined;
		if (columns && banner) {
			if (wide) {
				const centerWidth = width - sideWidth * 2 - 4;
				const middle = banner.render(centerWidth);
				const rows = Math.max(middle.length, columns.left.length, columns.right.length);
				const leftTop = Math.floor((rows - columns.left.length) / 2);
				const rightTop = Math.floor((rows - columns.right.length) / 2);
				for (let row = 0; row < rows; row++) {
					const left = truncateToWidth(columns.left[row - leftTop] ?? "", sideWidth);
					const mid = truncateToWidth(middle[row] ?? "", centerWidth);
					const right = truncateToWidth(columns.right[row - rightTop] ?? "", sideWidth);
					top.push(left + "\x1b[0m" + " ".repeat(sideWidth - visibleWidth(left) + 2) +
						mid + "\x1b[0m" + " ".repeat(centerWidth - visibleWidth(mid) + 2 + sideWidth - visibleWidth(right)) + right);
				}
			} else {
				const middle = banner.render(width);
				top.push(...(height < 24 ? middle.slice(-3) : middle));
				for (const group of [columns.left, columns.right]) {
					for (const row of wrapTextWithAnsi(group.join(" \x1b[0m· "), width)) top.push(centered(row, width));
				}
			}
			for (const row of columns.below) top.push(centered(row, width));
		}
		// Dialogs replace children in the editor container, never the editor object.
		for (const root of roots) {
			if (root === transcript || root === editor || (columns && (root === banner || root === status))) continue;
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
