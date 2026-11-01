# Project status (last updated 2026-11-01)

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

| protocol | design | bp formed | **unrolled IoU (graded)** | projected IoU | flatness | time |
|---|---|---|---|---|---|---|
| quick | original | 0.993 | 0.95 | 0.88 | 0.29 | 13 min |
| quick | twist-fixed | 0.993 | 0.95 | 0.90 | 0.19 | 12 min |
| shape | original | 0.994 | 0.95 | 0.88 | 0.27 | 22 min |
| shape | twist-fixed | 0.994 | 0.95 | 0.90 | 0.17 | 22 min |
| standard | original | 0.996 | 0.96 | **0.54** | 0.23 | 87 min |
| standard | twist-fixed | re-running with the crash fix (2026-11-01) | | | | |

Wrong-shape control (the same rectangles scored against a triangle): unrolled IoU ≈ 0.68.

**Key finding:** under the standard protocol (5M production steps) the flat rectangle **rolls up into a
tube** (end view: a ~22 nm ring) while staying fully base-paired. The shorter protocols ended before this
happened. Projected-outline IoU is therefore not a reliable flat-shape score in solution. **Resolved
2026-11-01:** shapes are now graded on the *unrolled* sheet (see the README), which is 0.95–0.96 on every
protocol, rolled or not. Projected IoU and flatness are still reported as in-solution descriptors. See
`runs/_calibration/protocols/rothemund_rect__standard/figures/structure_occupancy.png` (fourth panel).

Other calibration facts:
- The relaxed helix spacing is 2.64 nm, which validates the 2.6 nm footprint convention.
- Twist correction (1 deletion per 48 bp) cuts the end-to-end twist from 57° to 33°.
- scadnano's cadnano import loses the square lattice's 10.67 bp/turn and the helix phase. The Rothemund
  reference sets them (10.67 bp/turn with roll 305°; twist-fixed: 10.5 bp/turn with roll 320°).

## Open decisions (in rough priority order)

Resolved: flat-shape scoring now uses the unrolled sheet. Because that score is the same on every
protocol, the shorter `shape` grading protocol (~22 min per full-size design) is adequate. The standard
protocol only adds in-solution behaviour (rolling).

1. **Twist-fixed standard comparison**: re-running with the crash fix (results in
   `runs/_calibration/protocols_rerun/`). It shows whether twist correction prevents rolling.
2. **Harness policy:** should a model that stops without submitting get one reminder? Should there be a
   per-run token or cost cap? (DeepSeek used 5.1M tokens on one small task.)
3. **Where to run long experiments:** a rented cloud GPU is recommended. To do: a bootstrap script for a
   fresh Ubuntu GPU machine, `OXDNA_MODE=local`, resumable `compare_protocols.py`, and docs on copying
   `runs/` back.
4. Reference designs for the 4HB, 8-helix sheet, 12HB and shape tasks (positive controls).
5. Repeats: the pilot had n=1 per model. Use ≥3 repeats for real comparisons.
6. Choose a licence before making the repo public. The Rothemund cadnano files are CC BY-SA via
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
