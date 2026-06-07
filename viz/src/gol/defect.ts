// Temporal-emergence defect for Game of Life (port of holons/defect.py).
// Two coarse descriptions are compared at time tau:
//   actual    = coarseState(evolve(beta, tau))
//   predicted = Phi^+(coarseState(beta))  (catalogued objects translated)
// The symmetric-difference defect counts mismatched (pattern, anchor) keys;
// the matched defect grades displacement below creation/destruction.
import type { LifeField } from "./life";
import { step } from "./life";
import { coarseState, canonicalise } from "./coarse";
import type { PlacedObject } from "./coarse";
import { GLIDER_SE, GLIDER_NW, GLIDER_SE_VELOCITY, GLIDER_NW_VELOCITY } from "./patterns";
import { minCostAssignment } from "./matching";

// Catalogue: canonical pattern key -> per-period velocity.
const VELOCITIES = new Map<string, [number, number]>();
VELOCITIES.set(canonicalise(GLIDER_SE.map(([r, c]) => [r, c])).patternKey, GLIDER_SE_VELOCITY);
VELOCITIES.set(canonicalise(GLIDER_NW.map(([r, c]) => [r, c])).patternKey, GLIDER_NW_VELOCITY);

export interface PredictedObject {
  patternKey: string;
  anchor: [number, number];
}

// Phi^+: translate each catalogued object by velocity * nPeriods; uncatalogued
// objects are treated as stationary.
export function coarseEvolve(coarse: PlacedObject[], nPeriods: number): PredictedObject[] {
  return coarse.map((o) => {
    const v = VELOCITIES.get(o.patternKey) ?? [0, 0];
    return {
      patternKey: o.patternKey,
      anchor: [o.anchor[0] + v[0] * nPeriods, o.anchor[1] + v[1] * nPeriods],
    };
  });
}

function key(o: { patternKey: string; anchor: [number, number] }): string {
  return `${o.patternKey}@${o.anchor[0]},${o.anchor[1]}`;
}

export function evolve(f: LifeField, tau: number): LifeField {
  let r = f;
  for (let t = 0; t < tau; t++) r = step(r);
  return r;
}

export interface DefectResult {
  symdiff: number;
  matched: number;
  predicted: PredictedObject[];
  actual: PlacedObject[];
}

// Both defects at time tau (tau must be a multiple of period for phase alignment).
export function defectAt(beta: LifeField, tau: number, period: number): DefectResult {
  const nPeriods = Math.round(tau / period);
  const initial = coarseState(beta);
  const predicted = coarseEvolve(initial, nPeriods);
  const actual = coarseState(evolve(beta, tau));

  // Symmetric-difference defect over (pattern, anchor) keys.
  const predKeys = new Set(predicted.map(key));
  const actKeys = new Set(actual.map(key));
  let symdiff = 0;
  for (const k of predKeys) if (!actKeys.has(k)) symdiff++;
  for (const k of actKeys) if (!predKeys.has(k)) symdiff++;

  return { symdiff, matched: matchedDistance(predicted, actual), predicted, actual };
}

// Graded distance via min-cost matching (port of matched_distance).
export function matchedDistance(
  predicted: PredictedObject[],
  actual: Array<{ patternKey: string; anchor: [number, number] }>,
  unmatchedCost = 1.0,
  dispRate = 0.25,
  dispCap = 0.5,
): number {
  const m = predicted.length;
  const n = actual.length;
  if (m + n === 0) return 0;
  const big = (m + n) * unmatchedCost + 1.0;
  const size = m + n;
  const cost: number[][] = Array.from({ length: size }, () => new Array<number>(size).fill(big));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (predicted[i].patternKey === actual[j].patternKey) {
        const d =
          Math.abs(predicted[i].anchor[0] - actual[j].anchor[0]) +
          Math.abs(predicted[i].anchor[1] - actual[j].anchor[1]);
        cost[i][j] = Math.min(dispRate * d, dispCap);
      }
    }
    cost[i][n + i] = unmatchedCost; // predicted i destroyed
  }
  for (let j = 0; j < n; j++) {
    cost[m + j][j] = unmatchedCost; // actual j created
    for (let l = 0; l < m; l++) cost[m + j][n + l] = 0; // ghost filler
  }
  return minCostAssignment(cost);
}
