// Lenia explorer tab. Pick a preset organism (decoded from Chan's collection);
// watch it in single-organism mode (drag mu/sigma across the stability boundary)
// or in collision mode (actual A+B vs non-interacting prediction vs difference,
// with the live superposition defect).
import { presetParams, makeKernel, step, totalMass, centerOfMass } from "./lenia/lenia";
import type { LeniaParams, KernelEntry } from "./lenia/lenia";
import { PRESETS } from "./lenia/presets";
import type { LeniaPreset } from "./lenia/presets";
import { placeCentered } from "./lenia/field";
import { setupCollision, superpositionDefect, defectDensity } from "./lenia/collision";
import type { CollisionState } from "./lenia/collision";
import { GridCanvas } from "./ui/render";
import { LinePlot } from "./ui/plot";
import { el, button, slider, selectControl, metric } from "./ui/dom";

const N = 120;

export function createLeniaView(): HTMLElement {
  let preset: LeniaPreset = PRESETS[0];
  let params: LeniaParams = presetParams(preset);
  let kernel: KernelEntry[] = makeKernel(params);
  let mode: "single" | "collision" = "single";
  let separation = 45;
  let tau = 0;
  let playing = false;
  let stepsPerSec = 12;

  let collision: CollisionState = setupCollision(N, separation, preset.cells, kernel, params);
  const actualC = new GridCanvas(N, 230);
  const predC = new GridCanvas(N, 230);
  const diffC = new GridCanvas(N, 230);
  const deltaPlot = new LinePlot(360, 150, { xLabel: "tau" });
  deltaPlot.setSeries([{ key: "delta", color: "#e5484d", label: "δ superposition" }]);

  let single: Float32Array = placeCentered(N, preset.cells);
  let singleMass0 = totalMass(single);
  let prevCentroid: [number, number] = centerOfMass(single, N);
  const singleC = new GridCanvas(N, 360);

  const mTau = metric("tau (step)");
  const mDelta = metric("superposition δ", "temporal");
  const mMass = metric("mass ratio");
  const mSpeed = metric("speed (cells/step)");
  const mState = metric("state");
  const mInfo = metric("kernel / growth");

  const wrapD = (d: number) => ((d + N / 2) % N) - N / 2;

  function updateInfo(): void {
    mInfo.set(`R=${params.R} T=${Math.round(1 / params.dt)} rings=${params.b.length} `
      + `kn${params.kn} gn${params.gn}`);
  }

  function resetCollision(): void {
    collision = setupCollision(N, separation, preset.cells, kernel, params);
    tau = 0;
    deltaPlot.reset();
    drawCollision();
  }

  function resetSingle(): void {
    single = placeCentered(N, preset.cells);
    singleMass0 = totalMass(single);
    prevCentroid = centerOfMass(single, N);
    tau = 0;
    drawSingle();
  }

  function drawCollision(): void {
    actualC.drawScalar(collision.together, 1);
    const pred = new Float32Array(collision.together.length);
    for (let i = 0; i < pred.length; i++) pred[i] = collision.aAlone[i] + collision.bAlone[i];
    predC.drawScalar(pred, 1);
    diffC.drawScalar(defectDensity(collision), 0.4);
    const d = superpositionDefect(collision);
    mTau.set(String(tau));
    mDelta.set(d.toFixed(3));
    mMass.set((totalMass(collision.together) / collision.mass0).toFixed(2) + "×");
    mSpeed.set("—");
    mState.set(d < 1e-6 ? "independent" : "interacting");
    updateInfo();
  }

  function drawSingle(): void {
    singleC.drawScalar(single, 1);
    const c = centerOfMass(single, N);
    const speed = Math.hypot(wrapD(c[0] - prevCentroid[0]), wrapD(c[1] - prevCentroid[1]));
    prevCentroid = c;
    const ratio = totalMass(single) / singleMass0;
    mTau.set(String(tau));
    mDelta.set("—");
    mMass.set(ratio.toFixed(2) + "×");
    mSpeed.set(tau === 0 ? "—" : speed.toFixed(3));
    mState.set(ratio < 0.2 ? "dissolved" : ratio > 3 ? "exploding" : "alive");
    updateInfo();
  }

  function stepOnce(): void {
    if (mode === "collision") {
      collision.together = step(collision.together, N, kernel, params);
      collision.aAlone = step(collision.aAlone, N, kernel, params);
      collision.bAlone = step(collision.bAlone, N, kernel, params);
      tau += 1;
      deltaPlot.push(tau, { delta: superpositionDefect(collision) });
      deltaPlot.render();
      drawCollision();
    } else {
      single = step(single, N, kernel, params);
      tau += 1;
      drawSingle();
    }
  }

  let last = 0;
  let acc = 0;
  function loop(t: number): void {
    if (last === 0) last = t;
    acc += (t - last) / 1000;
    last = t;
    if (playing) {
      const interval = 1 / stepsPerSec;
      let did = 0;
      while (acc >= interval && did < 3) {
        stepOnce();
        acc -= interval;
        did++;
      }
    } else {
      acc = 0;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  const collisionGroup = el("div", { class: "panel" },
    el("h2", {}, "Lenia collision — actual vs non-interacting prediction vs difference"),
    el("div", { class: "canvas-row" },
      el("div", { class: "canvas-cell" }, el("div", { class: "label" }, "actual  evolve(A+B)"), actualC.canvas),
      el("div", { class: "canvas-cell" }, el("div", { class: "label" }, "predicted  evolve(A)+evolve(B)"), predC.canvas),
      el("div", { class: "canvas-cell" }, el("div", { class: "label" }, "difference  |actual − predicted|"), diffC.canvas),
    ),
    el("p", { class: "note" },
      "δ is the difference panel's total, mass-normalised: exactly 0 while the solitons are "
      + "independent (difference panel black), spiking at contact. Best with the single-ring gliders."),
    el("h2", { style: "margin-top:12px" }, "superposition defect δ(τ)"),
    deltaPlot.canvas,
  );

  const singleGroup = el("div", { class: "panel" },
    el("h2", {}, "Lenia single organism — drag μ, σ to cross the stability boundary"),
    singleC.canvas,
    el("p", { class: "note" },
      "Each preset glides, oscillates or rotates at its native μ, σ (mass ≈ conserved). "
      + "Move μ or σ away and it dissolves (mass → 0) or explodes — the stability boundary, live."),
  );

  const leftHost = el("div", {});

  function applyMode(m: "single" | "collision"): void {
    mode = m;
    leftHost.replaceChildren(m === "collision" ? collisionGroup : singleGroup);
    if (m === "collision") resetCollision();
    else resetSingle();
  }

  function selectPreset(idx: number): void {
    preset = PRESETS[idx];
    params = presetParams(preset);
    kernel = makeKernel(params);
    muSlider.set(params.mu);
    sigSlider.set(params.sigma);
    applyMode(mode);
  }

  const presetSel = selectControl(
    "preset organism",
    PRESETS.map((p, i) => ({ value: String(i), label: `${p.name} — ${p.kind}` })),
    "0",
    (v) => selectPreset(parseInt(v, 10)),
  );
  const modeSel = selectControl(
    "mode",
    [
      { value: "single", label: "single organism (stability)" },
      { value: "collision", label: "collision (two copies)" },
    ],
    "single",
    (v) => applyMode(v as "single" | "collision"),
  );
  const sepSlider = slider("separation", 20, 90, 1, separation, (v) => {
    separation = v;
    if (mode === "collision") resetCollision();
  });
  const muSlider = slider("μ growth centre", 0.0, 0.5, 0.002, params.mu, (v) => (params.mu = v), (v) => v.toFixed(3));
  const sigSlider = slider("σ growth width", 0.001, 0.06, 0.0005, params.sigma, (v) => (params.sigma = v), (v) => v.toFixed(4));
  const speedSlider = slider("speed (steps/s)", 1, 30, 1, stepsPerSec, (v) => (stepsPerSec = v));
  const playBtn = button("play", () => {
    playing = !playing;
    playBtn.textContent = playing ? "pause" : "play";
  }, true);

  const right = el("div", {},
    el("div", { class: "panel" },
      el("h2", {}, "controls"),
      el("div", { class: "controls" },
        presetSel,
        modeSel,
        sepSlider.el,
        muSlider.el,
        sigSlider.el,
        speedSlider.el,
        el("div", { class: "row" },
          playBtn,
          button("step", stepOnce),
          button("reset", () => applyMode(mode)),
        ),
      ),
    ),
    el("div", { class: "panel" },
      el("h2", {}, "live metrics"),
      el("div", { class: "metrics" }, mTau.el, mDelta.el, mMass.el, mSpeed.el, mState.el, mInfo.el),
    ),
  );

  updateInfo();
  applyMode("single");
  return el("div", { class: "layout" }, leftHost, right);
}
