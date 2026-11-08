# Project status (last updated 2026-11-02)

A snapshot for picking the work back up. The README covers how the harness works; this file covers
what has been run, what was found, and what is still open.

## What exists

- Shipped agent (`src/origami_agent/`): LangChain deep agent with scadnano design tools and design skills.
  No oxDNA; it checks a design (structurally) only when asked. Model-agnostic via `configs/models.yaml`.
- Internal benchmark (`src/synbio_bench/`): runs the shipped agent unchanged, adds `submit_design` and a run
  budget, and grades submissions with spec checks plus oxDNA. HTML reports with figures.
- Tasks: 4 core 3D tasks (`tasks/origami/`), 18 flat-shape × size tasks (`tasks/origami-shapes/`),
  2 Rothemund rectangles (`tasks/origami-rothemund/`), and a wireframe tetrahedron
  (`tasks/origami-wireframe/tetrahedron.yaml`).
- Reference designs (positive controls): the 6HB (`reference_designs/hb6_honeycomb.py`), the Rothemund
  rectangles from cadnano plus a `-twistfix` variant (`reference_designs/rothemund.py`), and a ~4000-nt
  Rothemund-style sharp triangle (`reference_designs/sharp_triangle.py`: three 7-helix trapezoids in helix
  groups rotated 0/120/240°, open centre, bridged seams), the reference for `origami-shapes/triangle-4000`;
  and a DAEDALUS-style wireframe tetrahedron (`reference_designs/wireframe_tetrahedron.py`: six 84-bp
  two-helix edges, each a helix group rotated in 3D, spanning-tree scaffold route, 1044 nt).
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

**Model comparison (`runs/origami-trio/`, `runs/triangle-domains/`, 1 run each, shape protocol, seed 1).**
Full write-up: the "Origami Trio Model Comparison" artifact (rebuilt from `runs/origami-trio/model_comparison.html`).

| task | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 | reference |
|---|---|---|---|---|
| rectangle-4000 (overall / unrolled IoU) | 0.961 / 0.89 | 0.970 / 0.92 | 0.973 / 0.93 | – |
| triangle-4000, raster prompt | 0.962 / 0.89 | 0.970 / 0.92 | 0.958 / 0.89 | – |
| triangle-4000, domains allowed | 0.959 / 0.89 (raster) | 0.964 / 0.90 (raster) | **0.977 / 0.94 (trapezoids)** | 0.977 / 0.95 |
| 6hb-honeycomb-3k (overall) | 0.9985 | 0.9983 | 0.9982 | – |
| wireframe tetrahedron (overall / wireframe F1, 3 seeds) | 0.836 / 0.51 | 0.916 / 0.75 | **0.939 / 0.83** | 0.931 / 0.80 |

- Scores barely separate the models; Haiku is ~10× cheaper per design ($0.04–0.08 vs $0.26–0.91).
- The first triangle prompt forced a single raster, which frays at the apex. With domains allowed and a
  sharp-corners section in the design-rules skill, only Opus switched to three bridged trapezoids; its
  triangle is the flattest of any run (flatness 0.19; reference 0.26; rasters 0.32–0.42).
- Wireframe tetrahedron (`runs/wireframe-tetrahedron/`): every model's design passed the spec and traced the
  target exactly, and all kept >99% of base pairs. In simulation, Sonnet and Opus (both following the skill's
  DAEDALUS recipe) hold the shape; Haiku used its own scaffold route with no vertex staples, and its corners hinge.
  Seed-to-seed spread is ~±0.05 F1 (reference 0.71–0.90 over five runs), so Opus vs reference is a tie and
  Opus vs Sonnet is suggestive only.
- Haiku uses sparser crossovers (rectangle 162 vs 234; 6HB 141 vs ~280) and its bundle is floppier
  (RMSD 2.2 vs 1.3 nm), which the 3D score doesn't see.
- Spend so far $5.85 by our tracking. LangSmith's price maps are wrong for Sonnet 5.5 cache reads ($0.10
  vs $0.20/M, ~15% low) and for Haiku 5.5 (~50% high); not changed (needs the user's OK).

**Calibration (`runs/_calibration/`), Rothemund rectangle, 13,824 nt** (unrolled IoU here is from the
first, MDS-only unroll; the current unroll reads ~0.02 higher on rectangles):

| protocol | design | bp formed | **unrolled IoU (graded)** | projected IoU | flatness | time |
|---|---|---|---|---|---|---|
| quick | original | 0.993 | 0.95 | 0.88 | 0.29 | 13 min |
| quick | twist-fixed | 0.993 | 0.95 | 0.90 | 0.19 | 12 min |
| shape | original | 0.994 | 0.95 | 0.88 | 0.27 | 22 min |
| shape | twist-fixed | 0.994 | 0.95 | 0.90 | 0.17 | 22 min |
| standard | original | 0.996 | 0.96 | **0.54** | 0.23 | 87 min |
| standard | twist-fixed | 0.996 | 0.96 | 0.61 | 0.27 | 85 min |

Wrong-shape control (the same rectangles scored against a triangle): unrolled IoU ≈ 0.68.

**Key finding:** under the standard protocol (5M production steps) the flat rectangle **rolls up into a
tube** (end view: a ~22 nm ring) while staying fully base-paired. The shorter protocols ended before this
happened. Twist correction does **not** prevent this: the twist-fixed rectangle also curls into an open
U (projected IoU 0.61 vs 0.54), so the curvature is not explained by global twist alone.
Projected-outline IoU is therefore not a reliable flat-shape score in solution. **Resolved 2026-11-01:** shapes are now graded on the *unrolled* sheet (see the README), which is 0.95–0.96 on every
protocol, rolled or not. Projected IoU and flatness are still reported as in-solution descriptors. See
`runs/_calibration/protocols/rothemund_rect__standard/figures/structure_occupancy.png` (fourth panel).

Other calibration facts:
- The relaxed helix spacing is 2.64 nm, which validates the 2.6 nm footprint convention.
- Twist correction (1 deletion per 48 bp) cuts the end-to-end twist from 57° to 33°.
- scadnano's cadnano import loses the square lattice's 10.67 bp/turn and the helix phase. The Rothemund
  reference sets them (10.67 bp/turn with roll 305°; twist-fixed: 10.5 bp/turn with roll 320°).

## Open decisions (in rough priority order)

Added 2026-11-02: 3D wireframe polyhedra. `wireframe:` in a task spec (type, edge_nm, tube_nm, min_f1) adds a
spec check on the design's own coordinates and a graded shape score: the oxDNA mean structure is fitted (ICP)
onto the ideal polyhedron's edges, and F1 combines precision (nucleotides within 3.5 nm of an edge) and recall
(edge interiors with DNA nearby) (`src/synbio_bench/origami/wireframe.py`; tetrahedron, cube, octahedron). The
score is noisy across oxDNA seeds, so the wireframe experiment grades with three seeds. Open: more wireframe
tasks (cube, octahedron, larger edges), and whether a longer protocol settles the bowing seen on every design.

Resolved 2026-11-02: multi-domain flat shapes. The shape prompts allow several domains (helix groups
rotated within the sheet's plane), joined by the scaffold and bridging staples. Grading changes:
`planar` means every helix axis lies in one plane; the footprint of multi-group designs is drawn from the
real helix axes; enclosed holes are filled before IoU; the unroll follows crossovers through short loops and
seams the design sets side by side, and refines the MDS layout against short 3D distances (on flat test
designs it matches the exact outline within 0.01; the old MDS-only unroll read raster triangles up to 0.02
high). All origami-trio flat runs were re-analysed. The grader also re-relaxes (up to 3 rounds) when a
backbone bond is still past oxDNA's FENE limit after relaxation; one Opus triangle failed production on a
stretched 10-T loop before this.

Resolved: flat-shape scoring now uses the unrolled sheet. Because that score is the same on every
protocol, the shorter `shape` grading protocol (~22 min per full-size design) is adequate. The standard
protocol only adds in-solution behaviour (rolling). The CUDA crash fix (`max_density_multiplier: 10`)
is confirmed on the 13,824-nt twist-fixed run.

Resolved 2026-11-02: oxDNA is benchmark-only. The agent no longer simulates or validates during runs
(it has a structural `check_design` it uses only on request), so scores measure the agent as shipped.
**This changes the task:** the pilot and calibration-era agents could simulate up to 4 times and validate
against the spec. Rerun the pilot before comparing models.

Resolved 2026-11-02: the stalled-run policy (README, "Run budget"). DeepSeek's pilot run never stopped on
its own: it spent all its calls hand-optimising staple routing and never built, validated or submitted a
design. Runs now get budget notices at 50/75/90%, one last chance to submit when the budget ends, and a
high cost cap as a safety net. The pilot predates this policy; rerun it before comparing models.

1. **Where to run long experiments:** draft Modal backend on branch `modal-oxdna-backend` (`mode: modal`,
   `docs/modal.md`): stages run on Modal GPUs, outputs come back through a Volume, analysis stays local, and seeds
   and runs can grade in parallel. It is tested only against a fake of the app. **Next:** sign up, deploy, run
   `scripts/modal_smoke.py`, then `scripts/modal_gpu_benchmark.py` (GPU type, verlet_skin, sims per GPU), and work through the checklist in `docs/modal.md` (image build, agreement with
   the local GTX 1660, oxDNA version pin, GPU choice by cost). For multi-hour standard-protocol runs, a rented
   GPU box with `OXDNA_MODE=local` is still the alternative (bootstrap script not written).
   **2026-11-05:** deployed and smoke-tested on Modal (6HB reference: bp 0.988). GPU benchmark in `docs/modal.md`:
   best value is an L4 with 3 sims sharing it under MPS and verlet_skin 0.1 (~$0.04 per shape-protocol seed for the
   13.8k-nt Rothemund, half the cost of 1 sim per GPU); A100s don't pay off; sharing without MPS is worse than serial.
   Done: the grader pools all runs in `evaluate`/`baseline` and runs simulations 3 to a GPU under MPS
   (`modal.sims_per_gpu`), similar design sizes together; default GPU L4 with A10G fallback. Checked on Modal with
   `scripts/modal_smoke.py --seeds 3` (3 seeds on one A10G with MPS, ~$0.037 per seed, results match a solo run). verlet_skin is now 0.1 for MD (MC relaxation keeps 0.5:
   oxDNA's MC needs > 0.17): re-grading origami-trio on Modal at 0.1 vs 0.5 kept every ranking, mean change -0.0015,
   largest -0.0085 (Opus triangle, one seed).
   **Next:** `configs/open-models.yaml`, open-weight frontier models plus an OpenAI model via OpenRouter on the
   origami-trio designs and the tetrahedron (3 seeds, set in the task file). Models chosen 2026-11-05: Kimi K3.1 Pro,
   GLM 5.3, Qwen3.8 Max, GPT-6.1 Sol, GPT-6.1 Terra, GPT-6 Luna (in configs/models.yaml, default reasoning effort).
   Before running: `scripts/estimate_costs.py` replays the Claude runs' per-call token usage at each model's prices. Modal billed $2.54 to date, about 30% over container-time estimates (image builds,
   warm containers).
2. Reference designs for the 4HB, 8-helix sheet, 12HB and the other shape tasks (positive controls). The
   star likely needs the same domain treatment as the triangle; `sharp_triangle.py` is fixed at ~4000 nt.
3. Repeats: the pilot had n=1 per model. Use ≥3 repeats for real comparisons.
4. Choose a licence before making the repo public. The Rothemund cadnano files are CC BY-SA via
   OpenWetware; see `reference_designs/cadnano/SOURCES.md`.

## Resuming

```bash
uv sync
uv run pytest                                   # 71 tests, no API keys, oxDNA or Modal needed
uv run synbio-bench --experiment configs/pilot.yaml report        # rebuild the pilot report
uv run synbio-bench --experiment configs/pilot.yaml report --embed
```

Local-only state that is not in git: `.env` (API keys), `runs/` (pilot, calibration and reports), and
`reference_designs/out/` (regenerate with the reference scripts).
