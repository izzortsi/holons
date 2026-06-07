// Game of Life (B3/S23) on a finite grid with an open (dead) boundary.
// Port of holons/life.py. Cells are a flat Uint8Array of 0/1, row-major.

export interface LifeField {
  w: number;
  h: number;
  cells: Uint8Array;
}

export function makeField(w: number, h: number): LifeField {
  return { w, h, cells: new Uint8Array(w * h) };
}

export function cloneField(f: LifeField): LifeField {
  return { w: f.w, h: f.h, cells: f.cells.slice() };
}

export function get(f: LifeField, r: number, c: number): number {
  if (r < 0 || r >= f.h || c < 0 || c >= f.w) return 0; // open boundary
  return f.cells[r * f.w + c];
}

// One B3/S23 update; returns a new field.
export function step(f: LifeField): LifeField {
  const out = makeField(f.w, f.h);
  for (let r = 0; r < f.h; r++) {
    for (let c = 0; c < f.w; c++) {
      let n = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          n += get(f, r + dr, c + dc);
        }
      }
      const alive = f.cells[r * f.w + c] === 1;
      const next = alive ? n === 2 || n === 3 : n === 3;
      out.cells[r * f.w + c] = next ? 1 : 0;
    }
  }
  return out;
}

export function totalLive(f: LifeField): number {
  let s = 0;
  for (let i = 0; i < f.cells.length; i++) s += f.cells[i];
  return s;
}
