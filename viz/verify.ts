// Headless check that the ported TS engines match the Python behaviour.
import { presetParams, makeKernel, step, totalMass, centerOfMass } from "./src/lenia/lenia";
import { PRESETS } from "./src/lenia/presets";
import { placeCentered } from "./src/lenia/field";
import { setupCollision, superpositionDefect } from "./src/lenia/collision";
import { makeField } from "./src/gol/life";
import { setupCollision as golCollision } from "./src/gol/patterns";
import { coarseState } from "./src/gol/coarse";
import { coarseEvolve, matchedDistance, evolve } from "./src/gol/defect";

let fail = 0;
function check(name: string, cond: boolean, info = ""): void {
  console.log((cond ? "OK   " : "FAIL ") + name + (info ? "  " + info : ""));
  if (!cond) fail++;
}

// 1. Every Lenia preset persists; gliders translate.
const n = 120;
for (const preset of PRESETS) {
  const p = presetParams(preset);
  const kernel = makeKernel(p);
  let field = placeCentered(n, preset.cells);
  const m0 = totalMass(field);
  const c0 = centerOfMass(field, n);
  for (let i = 0; i < 100; i++) field = step(field, n, kernel, p);
  const ratio = totalMass(field) / m0;
  const c1 = centerOfMass(field, n);
  const wrap = (d: number) => ((d + n / 2) % n) - n / 2;
  const dist = Math.hypot(wrap(c1[0] - c0[0]), wrap(c1[1] - c0[1]));
  check(`${preset.name} persists`, ratio > 0.5 && ratio < 2,
    `massRatio=${ratio.toFixed(2)} moved=${dist.toFixed(1)} (${preset.kind})`);
}

// 2. GoL collision defect: 0 before contact, >0 after.
const f = makeField(96, 96);
golCollision(f, 8);
const init = coarseState(f);
const golDefect = (tau: number) => matchedDistance(coarseEvolve(init, tau / 4), coarseState(evolve(f, tau)));
check("GoL free-flight δ=0", golDefect(8) === 0, `δ(8)=${golDefect(8)}`);
check("GoL post-contact δ>0", golDefect(40) > 0, `δ(40)=${golDefect(40)}`);

// 3. Lenia collision locality: separated Orbia => δ≈0.
const orb = PRESETS[0];
const op = presetParams(orb);
const ok = makeKernel(op);
const st = setupCollision(120, 60, orb.cells, ok, op);
for (let i = 0; i < 8; i++) {
  st.together = step(st.together, 120, ok, op);
  st.aAlone = step(st.aAlone, 120, ok, op);
  st.bAlone = step(st.bAlone, 120, ok, op);
}
const d = superpositionDefect(st);
check("Lenia separated δ≈0", d < 1e-6, `δ=${d.toExponential(1)}`);

console.log(fail ? `\n${fail} FAILED` : "\nall checks passed");
