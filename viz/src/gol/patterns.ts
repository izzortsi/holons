// Canonical Game of Life glider patterns (port of holons/patterns.py).
import type { LifeField } from "./life";

export type Cell = [number, number]; // [row, col]

// SE-moving glider, phase 0; returns to phase 0 shifted by (+1,+1) after 4 steps.
export const GLIDER_SE: Cell[] = [
  [0, 1], [1, 2], [2, 0], [2, 1], [2, 2],
];
export const GLIDER_SE_VELOCITY: Cell = [1, 1];

// NW-moving glider (180-degree rotation of SE); shifts by (-1,-1) after 4 steps.
export const GLIDER_NW: Cell[] = [
  [0, 0], [0, 1], [0, 2], [1, 0], [2, 1],
];
export const GLIDER_NW_VELOCITY: Cell = [-1, -1];

export const GLIDER_PERIOD = 4;

export function stamp(f: LifeField, cells: Cell[], anchor: Cell): void {
  const [r0, c0] = anchor;
  for (const [dr, dc] of cells) {
    const r = r0 + dr;
    const c = c0 + dc;
    if (r >= 0 && r < f.h && c >= 0 && c < f.w) f.cells[r * f.w + c] = 1;
  }
}

// Shared collision geometry (port of holons/geometry.py). Places an SE glider
// and an NW glider `separation` cells apart along the main diagonal so they
// head toward each other.
export const MARGIN = 5;

export function setupCollision(f: LifeField, separation: number): void {
  f.cells.fill(0);
  stamp(f, GLIDER_SE, [MARGIN, MARGIN]);
  stamp(f, GLIDER_NW, [MARGIN + 3 + separation, MARGIN + 3 + separation]);
}
