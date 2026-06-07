// Lenia organism collision via the superposition defect, for any preset.
// delta = ||evolve(A+B) - (evolve(A)+evolve(B))|| / mass: 0 while the organisms
// are separated, spiking at contact. B is the 180-degree rotation of A, so it
// glides along the reversed velocity; the velocity is measured per organism.
import { step, centerOfMass } from "./lenia";
import type { LeniaParams, KernelEntry } from "./lenia";
import { placeCentered, placeCentroid, rot180 } from "./field";

export interface CollisionState {
  together: Float32Array;
  aAlone: Float32Array;
  bAlone: Float32Array;
  mass0: number;
}

// Measure an organism's per-step velocity by evolving a lone copy.
export function measureVelocity(
  cells: number[][],
  n: number,
  kernel: KernelEntry[],
  p: LeniaParams,
  settle = 20,
  steps = 40,
): [number, number] {
  let field = placeCentered(n, cells);
  for (let i = 0; i < settle; i++) field = step(field, n, kernel, p);
  const c0 = centerOfMass(field, n);
  for (let i = 0; i < steps; i++) field = step(field, n, kernel, p);
  const c1 = centerOfMass(field, n);
  const wrap = (d: number) => ((d + n / 2) % n) - n / 2;
  return [wrap(c1[0] - c0[0]) / steps, wrap(c1[1] - c0[1]) / steps];
}

export function setupCollision(
  n: number,
  separation: number,
  cells: number[][],
  kernel: KernelEntry[],
  p: LeniaParams,
): CollisionState {
  const v = measureVelocity(cells, n, kernel, p);
  const speed = Math.hypot(v[0], v[1]);
  // Non-glider (rotator/oscillator): no glide axis, fall back to a diagonal.
  const vh: [number, number] =
    speed < 0.03 ? [Math.SQRT1_2, Math.SQRT1_2] : [v[0] / speed, v[1] / speed];
  const cr = n / 2;
  const cc = n / 2;
  const a = placeCentroid(n, cells, cr - (separation / 2) * vh[0], cc - (separation / 2) * vh[1]);
  const b = placeCentroid(n, rot180(cells), cr + (separation / 2) * vh[0], cc + (separation / 2) * vh[1]);
  let mass0 = 0;
  const together = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    mass0 += a[i] + b[i];
    together[i] = a[i] + b[i];
  }
  return { together, aAlone: a, bAlone: b, mass0 };
}

export function superpositionDefect(state: CollisionState): number {
  const { together, aAlone, bAlone, mass0 } = state;
  let s = 0;
  for (let i = 0; i < together.length; i++) s += Math.abs(together[i] - (aAlone[i] + bAlone[i]));
  return s / mass0;
}

export function defectDensity(state: CollisionState): Float32Array {
  const { together, aAlone, bAlone } = state;
  const out = new Float32Array(together.length);
  for (let i = 0; i < together.length; i++) out[i] = Math.abs(together[i] - (aAlone[i] + bAlone[i]));
  return out;
}
