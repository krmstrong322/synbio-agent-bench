# oxDNA on Modal (cloud GPUs)

`mode: modal` runs the grader's oxDNA simulations on [Modal](https://modal.com) instead of the local GPU.
The agent is not affected: oxDNA only runs at grading time.

**Status: draft, not yet run against a real Modal account.** The client side is tested against an
in-process fake of the deployed app (`tests/test_modal_runner.py`). The container image, and the timings
and costs below, still need a first real run (see the checklist).

## How it works

```
local (synbio-bench evaluate)                         Modal
-----------------------------                         -----
simulate(): export design, write input_<stage>
  for each stage (relax_mc, relax_md[, relax_md_k], prod):
    send the files the stage reads  ──────────────►  run_stage_cpu (relax_mc: oxDNA's MC is CPU-only)
    (input, topology, start conf, .op)                run_stage_gpu (MD stages, CUDA)
                                                        oxDNA <input> in a scratch dir
                                                        new/changed files ─► Volume jobs/<job>/
    ◄── exit code, logs, file list, GPU name ─────
    download outputs from the Volume, delete them
    local backbone check → maybe another relax round
analysis, scoring and figures (all local, unchanged)
```

- The app is in `src/synbio_bench/origami/modal_app.py`, and the client is in `src/synbio_bench/origami/modal_runner.py`.
  `oxdna._run_stage` hands stages to the client when `mode: modal`.
- The image is `nvidia/cuda:12.8.1-devel` with oxDNA **v3.7.0**, built with `CUDA_COMMON_ARCH=ON`, so it builds
  without a GPU and runs on any Modal GPU type.
- Outputs go through a Volume (`synbio-oxdna-runs`) because production trajectories can be roughly 100–200 MB
  for a full-M13 design. Each job's folder is deleted after download.
- Simulations run in parallel up to `modal.max_parallel` (default 4). Grading seeds run side by side, and
  `synbio-bench evaluate --parallel N` grades N runs at once. Locally, everything stays serialized as before.
- When Modal is unavailable (not installed, not logged in, or not deployed), `ModalUnavailable` is raised. That stops the
  evaluation instead of recording a failed simulation, so the run stays ungraded and can be retried.

## Setup (to do once you have an account)

1. Sign up at modal.com. The Starter plan includes monthly free credit (about $30/month at the time of writing).
2. Install the client and log in. The token is stored in `~/.modal.toml`, not in this repo:
   ```bash
   uv sync --extra modal
   uv run modal token new
   ```
3. Deploy the app. The first deploy builds the image and compiles oxDNA for every CUDA architecture, which takes
   a while. Later deploys reuse the cached image:
   ```bash
   uv run modal deploy src/synbio_bench/origami/modal_app.py
   # optional: OXDNA_MODAL_GPU=L4 uv run modal deploy ...   (the default GPU baked into the app)
   ```
4. Smoke test on the 6HB reference (quick protocol), then compare it with a local WSL run:
   ```bash
   uv run python scripts/modal_smoke.py
   OXDNA_MODE=wsl uv run python scripts/compare_protocols.py ...   # or any earlier local simulate() dir
   uv run python scripts/modal_smoke.py --compare <local work dir>
   ```
5. Grade with it:
   ```bash
   OXDNA_MODE=modal uv run synbio-bench --experiment configs/origami-trio.yaml evaluate --parallel 4
   ```
   Alternatively, set `runner.mode: modal` in `configs/oxdna.yaml`.

## Settings (`configs/oxdna.yaml` → `runner.modal`)

| key | default | meaning |
|---|---|---|
| `app` | `synbio-oxdna` | deployed app name |
| `volume` | `synbio-oxdna-runs` | Volume used to bring outputs back |
| `gpu` | `null` | GPU per call (`T4`, `L4`, `A10G`, `L40S`, ...); `null` uses the deployed default (T4). Env: `OXDNA_MODAL_GPU` |
| `max_parallel` | `4` | simulations at once (Starter allows about 10 GPUs) |
| `environment` | `null` | Modal environment, if you use more than one |

`timeout_s` (top level) also applies on Modal and is enforced inside the container. A single stage can run for
at most 6 h.

## Checklist for the first real run

- [ ] The image builds. If the CMake step fails, check whether oxDNA v3.7.0 still builds with CUDA 12.8 and
      try `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- [ ] `scripts/modal_smoke.py` passes, and the bp fraction and RMSF match a local run of the same design and
      seed within seed noise.
- [ ] Record the oxDNA version on the local WSL build. It tracks master, while Modal is pinned to v3.7.0. Rerun the
      reference designs on Modal before mixing local and Modal grades in one comparison.
- [ ] Time the `shape` protocol on a 4000-nt and a 13,824-nt design on T4 and L4. Pick the default GPU by
      cost per graded design, not speed.
- [ ] Check Modal's usage page after the smoke test, and compare it with the stage times in `log_*.txt`
      (`remote_seconds`).
- [ ] Decide whether to record Modal cost per run in `result.json`, next to the API cost.

## Open questions

- Is the per-stage design fast enough? Each stage pays for a container start and file transfer, which is
  small next to production MD, but noticeable for the `quick` protocol.
- Trajectory size: for very long runs, consider having oxDNA write fewer frames, or computing the mean structure
  remotely and returning only that.
