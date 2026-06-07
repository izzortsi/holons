// Game of Life explorer tab: live coarse-graining, temporal defect (predicted
// vs actual objects) and structural support |L|, with the dissociation plots.
import { makeField, cloneField, step } from "./gol/life";
import type { LifeField } from "./gol/life";
import { setupCollision, stamp, GLIDER_SE, GLIDER_PERIOD } from "./gol/patterns";
import { coarseState, centroid } from "./gol/coarse";
import type { PlacedObject } from "./gol/coarse";
import { coarseEvolve, matchedDistance } from "./gol/defect";
import { GridCanvas, componentColor } from "./ui/render";
import { LinePlot } from "./ui/plot";
import { el, button, slider, selectControl, metric } from "./ui/dom";

const N = 96;
const DISPLAY = 480;

function patternCentroidFromKey(key: string): [number, number] {
  const cells = key.split(";").map((s) => s.split(",").map(Number));
  let sr = 0, sc = 0;
  for (const [r, c] of cells) {
    sr += r;
    sc += c;
  }
  return [sr / cells.length, sc / cells.length];
}

export function createGolView(): HTMLElement {
  const grid = new GridCanvas(N, DISPLAY);

  let beta0: LifeField = makeField(N, N);
  let current: LifeField = makeField(N, N);
  let initialCoarse: PlacedObject[] = [];
  let tau = 0;
  let playing = false;
  let scenario = "collision";
  let separation = 8;

  const mTau = metric("tau (step)");
  const mObj = metric("coarse objects");
  const mSym = metric("temporal δ (symdiff)", "temporal");
  const mMatch = metric("temporal δ (matched)", "temporal");
  const mMinL = metric("structural min |L|", "structural");
  const mVerdict = metric("structurally emergent?");

  const tempPlot = new LinePlot(300, 150, { xLabel: "tau" });
  tempPlot.setSeries([
    { key: "sym", color: "#e5484d", label: "δ symdiff" },
    { key: "match", color: "#f0b429", label: "δ matched" },
  ]);
  const structPlot = new LinePlot(300, 150, { xLabel: "tau" });
  structPlot.setSeries([{ key: "minL", color: "#3b82f6", label: "min |L|" }]);

  function buildScenario(): void {
    beta0 = makeField(N, N);
    if (scenario === "collision") {
      setupCollision(beta0, separation);
    } else if (scenario === "single") {
      stamp(beta0, GLIDER_SE, [10, 10]);
    } else {
      // two parallel SE gliders, far apart: never interact.
      stamp(beta0, GLIDER_SE, [8, 8]);
      stamp(beta0, GLIDER_SE, [8, 50]);
    }
    initialCoarse = coarseState(beta0);
    current = cloneField(beta0);
    tau = 0;
    tempPlot.reset();
    structPlot.reset();
    measureAndDraw();
  }

  function measureAndDraw(): void {
    const actual = coarseState(current);
    mTau.set(String(tau));
    mObj.set(String(actual.length));

    // Structural support |L| (component sizes).
    const sizes = actual.map((o) => o.size);
    const minL = sizes.length ? Math.min(...sizes) : 0;
    mMinL.set(sizes.length ? String(minL) : "—");
    const emergent = sizes.length > 0 && minL >= 2;
    mVerdict.set(emergent ? "yes (|L| ≥ 2)" : sizes.length ? "no (bare cell)" : "no objects");

    let predicted: ReturnType<typeof coarseEvolve> = [];
    if (tau % GLIDER_PERIOD === 0) {
      predicted = coarseEvolve(initialCoarse, tau / GLIDER_PERIOD);
      const predKeys = new Set(predicted.map((o) => `${o.patternKey}@${o.anchor[0]},${o.anchor[1]}`));
      const actKeys = new Set(actual.map((o) => `${o.patternKey}@${o.anchor[0]},${o.anchor[1]}`));
      let sym = 0;
      for (const k of predKeys) if (!actKeys.has(k)) sym++;
      for (const k of actKeys) if (!predKeys.has(k)) sym++;
      const matched = matchedDistance(predicted, actual);
      mSym.set(String(sym));
      mMatch.set(matched.toFixed(2));
      tempPlot.push(tau, { sym, match: matched });
      structPlot.push(tau, { minL });
      tempPlot.render();
      structPlot.render();
    }

    // Draw grid with component tint + centroids; ghost predicted positions.
    const compIndex = new Int32Array(N * N).fill(-1);
    actual.forEach((o, i) => {
      for (const [r, c] of o.cells) compIndex[r * N + c] = i;
    });
    grid.drawLife(current.cells, compIndex);
    actual.forEach((o, i) => {
      const [cr, cc] = centroid(o);
      grid.marker(cr, cc, componentColor(i), 4);
    });
    for (const p of predicted) {
      const [pr, pc] = patternCentroidFromKey(p.patternKey);
      grid.ring(p.anchor[0] + pr, p.anchor[1] + pc, "rgba(255,255,255,0.8)", 6);
    }
  }

  function stepOnce(): void {
    current = step(current);
    tau += 1;
    measureAndDraw();
  }

  // Animation loop.
  let last = 0;
  let acc = 0;
  let stepsPerSec = 8;
  function loop(t: number): void {
    if (last === 0) last = t;
    const dt = (t - last) / 1000;
    last = t;
    if (playing) {
      acc += dt;
      const interval = 1 / stepsPerSec;
      let did = 0;
      while (acc >= interval && did < 6) {
        stepOnce();
        acc -= interval;
        did++;
      }
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // Controls.
  const sepSlider = slider("separation", 5, 40, 1, separation, (v) => {
    separation = v;
    if (scenario === "collision") buildScenario();
  });
  const scenarioSel = selectControl(
    "scenario",
    [
      { value: "collision", label: "glider collision" },
      { value: "single", label: "single glider" },
      { value: "parallel", label: "two parallel (no contact)" },
    ],
    scenario,
    (v) => {
      scenario = v;
      buildScenario();
    },
  );
  const speedSlider = slider("speed (steps/s)", 1, 30, 1, stepsPerSec, (v) => (stepsPerSec = v));
  const playBtn = button("play", () => {
    playing = !playing;
    playBtn.textContent = playing ? "pause" : "play";
  }, true);

  const left = el("div", { class: "panel" },
    el("h2", {}, "Game of Life — actual grid, coarse objects, predicted ghosts"),
    grid.canvas,
    el("div", { class: "legend" },
      el("span", {}, el("span", { class: "swatch", style: "background:#d6dde6" }), "live cell"),
      el("span", {}, "● actual object centroid"),
      el("span", {}, "○ predicted Φ⁺ position"),
    ),
    el("p", { class: "note" },
      "Cells are tinted by Moore-connected component. ● marks each actual object's centroid; ",
      "○ marks where Φ⁺ predicts it (catalogued glider translated by its velocity). ",
      "Their mismatch is the temporal defect; |L| (cells per object) is the structural support."),
  );

  const right = el("div", {},
    el("div", { class: "panel" },
      el("h2", {}, "controls"),
      el("div", { class: "controls" },
        scenarioSel,
        sepSlider.el,
        speedSlider.el,
        el("div", { class: "row" },
          playBtn,
          button("step", stepOnce),
          button("reset", buildScenario),
        ),
      ),
    ),
    el("div", { class: "panel" },
      el("h2", {}, "live metrics (defect at τ multiples of 4)"),
      el("div", { class: "metrics" },
        mTau.el, mObj.el, mSym.el, mMatch.el, mMinL.el, mVerdict.el),
    ),
    el("div", { class: "panel" },
      el("h2", {}, "temporal defect δ(τ)"),
      tempPlot.canvas,
      el("h2", { style: "margin-top:12px" }, "structural support min |L|(τ)"),
      structPlot.canvas,
      el("p", { class: "note" },
        "The dissociation: structural |L| stays ≥ 2 (emergent throughout) while the temporal "
        + "defect is 0 in free flight and spikes at the collision."),
    ),
  );

  buildScenario();
  return el("div", { class: "layout" }, left, right);
}
