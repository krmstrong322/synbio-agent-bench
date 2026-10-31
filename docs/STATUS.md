# Project status: where we left off (2026-10-31)

A snapshot for picking the work back up. The README covers how the harness works; this file covers
what has been run, what was found, and what is still open.

## What exists

- Harness: LangChain deep agent with scadnano design tools, an oxDNA simulation tool, skills, a grader,
  and HTML reports with figures. It is model-agnostic, configured in `configs/models.yaml`.
- Tasks: 4 core 3D tasks (`tasks/origami/`), 18 flat-shape × size tasks (`tasks/origami-shapes/`), and
  2 Rothemund rectangles (`tasks/origami-rothemund/`).
- Reference designs (positive controls): the 6HB (`reference_designs/hb6_honeycomb.py`) and the Rothemund
  rectangles from cadnano, plus a `-twistfix` variant (`reference_designs/rothemund.py`).
- oxDNA runs in WSL Ubuntu on the GTX 1660 (CUDA), at `/usr/local/bin/oxDNA`.
- Cost tracking is cache-aware. It was validated within 3.5% of OpenRouter's actual bill.

## Results so far (local `runs/`, not in git)

**Pilot (`runs/origami-pilot/`, 4HB square task, 1 run each):**

| model | ended | overall | bp formed | cost |
|---|---|---|---|---|
| claude-haiku-5-5 | completed in 6 min, 13 calls | 1.00 | 0.99 | $0.021 |
| deepseek-v4.1-flash | hit the step limit after 48 min and 133 calls; never validated or submitted | 0.00 | – | $0.238 |

An earlier DeepSeek attempt (archived in `runs/origami-pilot/_deepseek_rep0_16k_output_limit/`) used its
whole 16k output cap on reasoning and stopped. The cap is now 64k.

**Calibration (`runs/_calibration/`), Rothemund rectangle, 13,824 nt:**

| protocol | design | bp formed | shape IoU | flatness | time |
|---|---|---|---|---|---|
| quick | original | 0.993 | 0.88 | 0.29 | 13 min |
| quick | twist-fixed | 0.993 | 0.90 | 0.19 | 12 min |
| shape | original | 0.994 | 0.88 | 0.27 | 22 min |
| shape | twist-fixed | 0.994 | 0.90 | 0.17 | 22 min |
| standard | original | 0.996 | **0.54** | 0.23 | 87 min |
| standard | twist-fixed | crashed (CUDA cell overflow); fixed since, not yet re-run | | | |

**Key finding:** under the standard protocol (5M production steps) the flat rectangle **rolls up into a
tube** (end view: a ~22 nm ring) while staying fully base-paired. The shorter protocols ended before this
happened, so their shape scores are not converged; they capture an early state. Projected-outline IoU
is therefore not a reliable flat-shape score in solution. See
`runs/_calibration/protocols/rothemund_rect__standard/figures/structure_occupancy.png`.

Other calibration facts:
- The relaxed helix spacing is 2.64 nm, which validates the 2.6 nm footprint convention.
- Twist correction (1 deletion per 48 bp) cuts the end-to-end twist from 57° to 33°.
- scadnano's cadnano import loses the square lattice's 10.67 bp/turn and the helix phase. The Rothemund
  reference sets them (10.67 bp/turn with roll 305°; twist-fixed: 10.5 bp/turn with roll 320°).

## Uncommitted when saved → now committed

- `oxdna.py`: `max_density_multiplier = 10` on CUDA, to fix the cell-overflow crash. **The test suite was
  not re-run after this change**, so run `uv run pytest` first.

## Open decisions (in rough priority order)

1. **How to score flat shapes, given that sheets roll up in solution.** Options:
   - *Developed-surface IoU (recommended):* unroll the simulated sheet (geodesic distances along the
     helix-axis graph, embedded in 2D by MDS) and compare that outline with the target. This is
     invariant to rolling and bending. Keep projected IoU and flatness as separate "in-solution shape"
     metrics.
   - *Adsorption plane:* add an attractive plane (as on AFM mica) so sheets stay flat. This is closer to
     experiment but adds physics assumptions; check that oxDNA's external forces support it.
   - Keep the projected IoU but use a long protocol (expensive, and rolling dominates).
2. **Shape-task grading protocol.** The shorter `shape` protocol was assumed to be converged; the standard
   run showed it isn't. Revisit after deciding (1).
3. **Re-run the twist-fixed standard comparison** with the crash fix (~95 min of GPU time):
   `uv run python scripts/compare_protocols.py --task origami-rothemund/rectangle --design reference_designs/out/rothemund_rect_twistfix.sc --protocol standard`
   (first rebuild the designs with `reference_designs/rothemund.py`, since `reference_designs/out/` is not in git).
4. **Harness policy:** should a model that stops without submitting get one reminder? Should there be a
   per-run token or cost cap? (DeepSeek used 5.1M tokens on one small task.)
5. **Where to run long experiments:** a rented cloud GPU is recommended. To do: a bootstrap script for a
   fresh Ubuntu GPU machine, `OXDNA_MODE=local`, resumable `compare_protocols.py`, and docs on copying
   `runs/` back.
6. Reference designs for the 4HB, 8-helix sheet, 12HB and shape tasks (positive controls).
7. Repeats: the pilot had n=1 per model. Use ≥3 repeats for real comparisons.
8. Choose a licence before making the repo public. The Rothemund cadnano files are CC BY-SA via
   OpenWetware; see `reference_designs/cadnano/SOURCES.md`.

## Resuming

```bash
uv sync
uv run pytest                                   # 18 tests, no API keys or oxDNA needed
uv run synbio-bench --experiment configs/pilot.yaml report        # rebuild the pilot report
uv run synbio-bench --experiment configs/pilot.yaml report --embed
```

Local-only state that is not in git: `.env` (API keys), `runs/` (pilot, calibration and reports), and
`reference_designs/out/` (regenerate with the reference scripts).
