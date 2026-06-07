// Coarse-graining: liveness grid -> placed objects (port of holons/coarse.py).
// A placed object is a Moore-connected component, canonicalised to a
// translation-invariant pattern key plus an anchor (min-corner).
import type { LifeField } from "./life";

export interface PlacedObject {
  patternKey: string; // canonical cells, sorted, as a string
  anchor: [number, number]; // (minRow, minCol)
  cells: Array<[number, number]>; // absolute cells of this component
  size: number; // |L|: live-cell count == structural support breadth
}

// Moore-connected (8-neighbour) components of live cells.
export function mooreComponents(f: LifeField): Array<Array<[number, number]>> {
  const visited = new Uint8Array(f.w * f.h);
  const comps: Array<Array<[number, number]>> = [];
  for (let r0 = 0; r0 < f.h; r0++) {
    for (let c0 = 0; c0 < f.w; c0++) {
      const idx0 = r0 * f.w + c0;
      if (f.cells[idx0] === 0 || visited[idx0]) continue;
      const comp: Array<[number, number]> = [];
      const stack: Array<[number, number]> = [[r0, c0]];
      while (stack.length > 0) {
        const [r, c] = stack.pop()!;
        if (r < 0 || r >= f.h || c < 0 || c >= f.w) continue;
        const idx = r * f.w + c;
        if (visited[idx] || f.cells[idx] === 0) continue;
        visited[idx] = 1;
        comp.push([r, c]);
        for (let dr = -1; dr <= 1; dr++)
          for (let dc = -1; dc <= 1; dc++)
            if (dr !== 0 || dc !== 0) stack.push([r + dr, c + dc]);
      }
      comps.push(comp);
    }
  }
  return comps;
}

export function canonicalise(cells: Array<[number, number]>): PlacedObject {
  let minR = Infinity;
  let minC = Infinity;
  for (const [r, c] of cells) {
    if (r < minR) minR = r;
    if (c < minC) minC = c;
  }
  const rel = cells
    .map(([r, c]): [number, number] => [r - minR, c - minC])
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const patternKey = rel.map(([r, c]) => `${r},${c}`).join(";");
  return { patternKey, anchor: [minR, minC], cells, size: cells.length };
}

export function coarseState(f: LifeField): PlacedObject[] {
  return mooreComponents(f).map(canonicalise);
}

// Centroid of a component (for display).
export function centroid(obj: PlacedObject): [number, number] {
  let sr = 0;
  let sc = 0;
  for (const [r, c] of obj.cells) {
    sr += r;
    sc += c;
  }
  return [sr / obj.cells.length, sc / obj.cells.length];
}
