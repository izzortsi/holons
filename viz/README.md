# Holons Explorer

Interactive TypeScript suite for visually inspecting the framework's **temporal**
and **structural** emergence defects on Game of Life and Lenia — the same
metrics the Python experiments compute (the engines are direct ports, verified
to match: see `npm run verify`).

## Run

```bash
cd viz
npm install
npm run dev      # open the printed http://localhost:5173
```

Other scripts:

- `npm run build` — type-check (`tsc --noEmit`) and bundle.
- `npm run verify` — headless check that the ported engines match the Python
  (Orbium glides with mass ratio 0.93 / 24.5 cells, GoL post-contact δ=3,
  Lenia separated δ=0).

## What it shows

### Game of Life tab
The live grid, tinted by Moore-connected component (the coarse-graining). For
each coarse object a filled dot marks its **actual** centroid; a hollow ring
marks where **Φ⁺** (the catalogued glider translated by its velocity) predicts
it. The mismatch is the **temporal defect** — shown both as the symmetric-
difference count and the matched (graded) distance. The **structural support**
`min |L|` (cells per object) reads out alongside. The two live plots build the
dissociation as it runs: structural `|L|` stays ≥ 2 (emergent throughout) while
the temporal defect is 0 in free flight and spikes at the collision.

Scenarios: glider collision (with separation slider), single glider (δ stays 0),
two parallel gliders (never interact).

### Lenia tab
A **preset selector** offers 8 organisms decoded from Bert Chan's collection
(`src/lenia/presets.ts`): single-ring gliders (Orbium, Discutium, Synorbium),
multi-ring gliders (Hydrogeminium, Scutium, Kronium), an oscillator (Helicium)
and a rotator (Nonadentium). The engine uses Chan's exact polynomial kernel/
growth cores and multi-ring kernels, so each reproduces faithfully
(`npm run verify` evolves all 8).

- **Collision mode**: three fields side by side — the actual joint evolution
  `evolve(A+B)`, the non-interacting prediction `evolve(A)+evolve(B)`, and their
  difference. The **superposition defect** δ is the difference panel's total
  (mass-normalised): exactly 0 while the difference panel is black (independent
  solitons), spiking at contact. The live δ(τ) curve plots it.
- **Single-organism mode**: drag the **μ** and **σ** sliders to cross the
  stability boundary live — the Orbium glides at μ=0.15, σ=0.014 (mass ≈
  conserved, steady speed) and dissolves or explodes away from it.

## Notes

- Pure browser, no backend. Lenia uses direct toroidal convolution (no FFT
  dependency); grids are 96×96 for interactive rates.
- The metric code under `src/gol/` and `src/lenia/` is a faithful port of the
  Python in `../holons/`; `verify.ts` asserts they agree numerically.
