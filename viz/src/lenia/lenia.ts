// Lenia engine with Bert Chan's exact kernel/growth cores, multi-ring kernels,
// and direct toroidal convolution (no FFT dependency). Faithful to
// github.com/Chakazul/Lenia so the decoded presets reproduce correctly.
import type { LeniaPreset } from "./presets";

export interface LeniaParams {
  R: number; // kernel radius (cells)
  dt: number; // time step = 1/T
  mu: number; // growth centre m
  sigma: number; // growth width s
  b: number[]; // kernel ring weights
  kn: number; // kernel core id (1 = polynomial, 2 = exp bump)
  gn: number; // growth core id (1 = polynomial, 2 = gaussian)
}

export interface KernelEntry {
  dr: number;
  dc: number;
  w: number;
}

export function presetParams(p: LeniaPreset): LeniaParams {
  return { R: p.R, dt: 1 / p.T, mu: p.m, sigma: p.s, b: p.b, kn: p.kn, gn: p.gn };
}

// Kernel core, normalised radius r in [0,1].
function kernelCore(kn: number, r: number): number {
  if (kn === 2) return Math.exp(4 - 1 / (r * (1 - r) + 1e-9)); // exp bump
  const t = 4 * r * (1 - r); // polynomial (4r(1-r))^4
  return t * t * t * t;
}

// Growth core, potential u.
function growthCore(gn: number, u: number, m: number, s: number): number {
  if (gn === 2) return 2 * Math.exp(-((u - m) * (u - m)) / (2 * s * s)) - 1; // gaussian
  const t = Math.max(0, 1 - ((u - m) * (u - m)) / (9 * s * s)); // polynomial
  return 2 * t * t * t * t - 1;
}

// Precompute the multi-ring kernel as a sparse offset list (Chan's calc_kernel:
// ring index = floor(B*r), within-ring radius = (B*r) mod 1, weighted by b[ring]).
export function makeKernel(p: LeniaParams): KernelEntry[] {
  const R = p.R;
  const B = p.b.length;
  const raw: KernelEntry[] = [];
  let total = 0;
  let maxW = 0;
  for (let dr = -R; dr <= R; dr++) {
    for (let dc = -R; dc <= R; dc++) {
      const D = Math.sqrt(dr * dr + dc * dc) / R;
      if (D >= 1) continue;
      const Br = B * D;
      const ring = Math.min(Math.floor(Br), B - 1);
      const w = kernelCore(p.kn, Math.min(Br % 1, 1)) * p.b[ring];
      if (w <= 0) continue;
      raw.push({ dr, dc, w });
      total += w;
      if (w > maxW) maxW = w;
    }
  }
  const cutoff = maxW * 1e-3; // drop near-zero tails to shrink the convolution
  const kernel: KernelEntry[] = [];
  for (const e of raw) if (e.w > cutoff) kernel.push({ dr: e.dr, dc: e.dc, w: e.w / total });
  return kernel;
}

export function step(
  field: Float32Array,
  n: number,
  kernel: KernelEntry[],
  p: LeniaParams,
): Float32Array {
  const R = p.R;
  const P = n + 2 * R;
  const padded = new Float32Array(P * P);
  for (let r = 0; r < P; r++) {
    const sr = (((r - R) % n) + n) % n;
    for (let c = 0; c < P; c++) {
      const sc = (((c - R) % n) + n) % n;
      padded[r * P + c] = field[sr * n + sc];
    }
  }
  const out = new Float32Array(n * n);
  const { mu, sigma, dt, gn } = p;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let u = 0;
      const base = (r + R) * P + (c + R);
      for (let k = 0; k < kernel.length; k++) {
        const e = kernel[k];
        u += e.w * padded[base + e.dr * P + e.dc];
      }
      let v = field[r * n + c] + dt * growthCore(gn, u, mu, sigma);
      if (v < 0) v = 0;
      else if (v > 1) v = 1;
      out[r * n + c] = v;
    }
  }
  return out;
}

export function totalMass(field: Float32Array): number {
  let s = 0;
  for (let i = 0; i < field.length; i++) s += field[i];
  return s;
}

export function centerOfMass(field: Float32Array, n: number): [number, number] {
  let mass = 0;
  const rowMarg = new Float64Array(n);
  const colMarg = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = field[r * n + c];
      rowMarg[r] += v;
      colMarg[c] += v;
      mass += v;
    }
  }
  if (mass === 0) return [0, 0];
  return [circularMean(rowMarg, n), circularMean(colMarg, n)];
}

function circularMean(marginal: Float64Array, n: number): number {
  let cs = 0, sn = 0;
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / n;
    cs += marginal[i] * Math.cos(a);
    sn += marginal[i] * Math.sin(a);
  }
  let ang = Math.atan2(sn, cs);
  if (ang < 0) ang += 2 * Math.PI;
  return (ang * n) / (2 * Math.PI);
}
