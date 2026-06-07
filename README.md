# Holons: simulations for an unbounded dynamical holarchy framework

Computer experiments for the framework defined in "Complex systems as unbounded
dynamical holarchies" (version 17). The framework localises emergence at
coarse-grainings and distinguishes structural and temporal emergence; this
repository measures the temporal-emergence defect δ^t_τ on substrates where
the verdict is checkable.

## What is here

The first experiment is the Conway's Game of Life glider collision spectrum,
following Section 7 of the framework. Further experiments (Lenia stability
boundary, 2D Ising RG critical scaling, Klein–Hoel causal-emergence
comparison, particle-life organisms) are real research projects, not weekend
scripts, and are not implemented here.

```
holons/
├── holons/           # the library
│   ├── life.py       # B3/S23 cellular automaton (pure numpy)
│   ├── patterns.py   # SE/NW glider phase-0 patterns, period, velocity
│   ├── coarse.py     # Moore-connected components + canonical (pattern, anchor)
│   └── defect.py     # δ^t_τ computation
├── scripts/
│   ├── sanity_check.py        # verify framework invariants on simple configs
│   └── collision_spectrum.py  # experiment 1: glider-glider defect spectrum
└── pyproject.toml    # numpy is the only required dependency
```

## Quick start

```bash
cd /home/istrozzi/Documents/kk-digital-docker/holons
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Sanity check: glider returns to itself shifted by (1,1) after 4 steps,
# isolated glider has zero defect, two parallel gliders never collide.
python3 scripts/sanity_check.py

# First experiment: defect spectrum over (separation, tau).
mkdir -p results
python3 scripts/collision_spectrum.py \
    --output-csv results/collision_spectrum.csv \
    --grid-size 80 \
    --tau-max 80 \
    --separation-min 5 \
    --separation-max 25
```

Output CSV columns: `separation, tau, defect`. Plot with whatever tool you
prefer.

## Visualising a collision

To see the GoL evolution synchronised with the defect heatmap for a chosen
separation, install the optional visualisation dependencies and produce an
animated GIF:

```bash
pip install -e ".[viz]"

python3 scripts/animate_collision.py \
    --csv results/collision_spectrum.csv \
    --separation 8 \
    --grid-size 80 \
    --tau-max 80 \
    --output results/collision_separation_08.gif
```

Each frame advances the GoL by one fine step and moves a red cursor across
the heatmap. The defect annotation in the title updates at every multiple
of the glider period (4); on intermediate steps it shows the last measured
value.

Separations 7 (odd) and 8 (even) are good first choices because they sit
on opposite sides of the parity-dependent plateau in the spectrum
(annihilation versus block production). Pass any separation present in
the CSV. To produce an MP4 instead of a GIF, end `--output` in `.mp4`
(requires `ffmpeg` on PATH). To show the full grid instead of the
auto-cropped action region, pass `--no-crop`.

## What the experiment measures

For each (separation, tau) pair:

1. Place an SE-moving glider at a fixed anchor and an NW-moving glider at
   the SE anchor offset by `(separation, separation)` so they head toward
   each other along the diagonal.
2. Evolve the fine dynamics (B3/S23) for `tau` steps. `tau` is constrained
   to be a multiple of the glider period (4), so each glider's phase
   returns to phase 0 at each measurement.
3. Coarse-grain the result: extract Moore-connected components, canonicalise
   each as `(pattern, anchor)` where `pattern` is translation-canonical
   (the frozen set of cells shifted so the minimum corner is `(0, 0)`) and
   `anchor` is the minimum corner of the bounding box.
4. Predict the coarse state by Φ^+: translate the initial coarse objects by
   their catalogued velocities times the number of periods elapsed.
5. Defect = cardinality of symmetric difference between actual and
   predicted coarse states.

The expected qualitative structure: `δ = 0` for τ less than the
time-to-contact (gliders haven't met), `δ > 0` after contact (the reaction
produces a configuration different from "two translated gliders"). The
interesting question is whether `δ` depends on separation in a structured
way — which would mean the defect is informative about the dynamics, not
just a binary "did anything happen" indicator. That is the framework's
first computable claim.

## Honest limitations of this first cut

- **τ multiples of period only.** At intermediate τ the glider is in a
  different phase and the §7 simplification "advance by (1,1) per period"
  doesn't directly apply. A phase-aware extension catalogues each phase
  and its per-step velocity; straightforward, not yet implemented.
- **Catalogue is implicit.** Any Moore-connected component is canonicalised
  on the fly. Reaction products with patterns not in the velocities map
  (still lifes, oscillators, new spaceships) are treated as stationary
  per §7. Extending the catalogue is just adding entries to the velocities
  map in `collision_spectrum.py`.
- **Open boundary.** For long τ, gliders may reach the grid boundary.
  Choose `--grid-size` large enough that this doesn't happen; the script
  warns when the chosen size looks too small.
- **One experiment.** This is the smallest credible test of the framework's
  defect being informative on a substrate with known microscopic rules.
  If the spectrum has structure, the framework has earned the next
  experiment.

## Reproducibility

All scripts accept CLI arguments — there are no hardcoded paths, grid sizes,
or experiment parameters in the source files. Output paths are
caller-controlled. To reproduce a specific result, record the full command
line in your experiment log.

## Extending

To add a new spaceship (e.g., LWSS, MWSS, HWSS):

1. Add its phase-0 canonical pattern and per-period velocity in
   `holons/patterns.py`.
2. Add the (pattern, velocity) pair to the `velocities` dict in the
   experiment script (or pass it through as a config file).
3. Update `sanity_check.py` to verify the new pattern's periodicity.

The framework's `defect` function does not care which patterns are in the
catalogue — it just looks them up. The same code handles a glider collision
or a Gosper gun firing into a Herschel track, as long as the velocities
dict knows the canonical translations.
