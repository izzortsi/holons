// Field placement helpers (independent of any specific organism).

export function rot180(cells: number[][]): number[][] {
  return cells.map((row) => [...row].reverse()).reverse();
}

function cellsCentroid(cells: number[][]): [number, number] {
  let mass = 0, sr = 0, sc = 0;
  for (let r = 0; r < cells.length; r++) {
    for (let c = 0; c < cells[r].length; c++) {
      const v = cells[r][c];
      mass += v;
      sr += v * r;
      sc += v * c;
    }
  }
  return mass === 0 ? [0, 0] : [sr / mass, sc / mass];
}

// Place `cells` into a fresh n-by-n torus field with its centroid at target.
export function placeCentroid(
  n: number,
  cells: number[][],
  targetR: number,
  targetC: number,
): Float32Array {
  const [cmR, cmC] = cellsCentroid(cells);
  const r0 = Math.round(targetR - cmR);
  const c0 = Math.round(targetC - cmC);
  const field = new Float32Array(n * n);
  for (let r = 0; r < cells.length; r++) {
    for (let c = 0; c < cells[r].length; c++) {
      const v = cells[r][c];
      if (v === 0) continue;
      const rr = (((r0 + r) % n) + n) % n;
      const cc = (((c0 + c) % n) + n) % n;
      field[rr * n + cc] = v;
    }
  }
  return field;
}

export function placeCentered(n: number, cells: number[][]): Float32Array {
  return placeCentroid(n, cells, n / 2, n / 2);
}
