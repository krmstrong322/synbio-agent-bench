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

## Pricing (modal.com/pricing, read 2026-11-03)

Billed per second. Starter plan: $0 plus **$30/month free credit**, 100 containers, **10 GPUs at once**.

| GPU | $/s | ≈ $/h |
|---|---|---|
| T4 | 0.000164 | 0.59 |
| L4 | 0.000222 | 0.80 |
| A10 | 0.000306 | 1.10 |
| L40S | 0.000542 | 1.95 |
| A100 40 GB / 80 GB | 0.000583 / 0.000694 | 2.10 / 2.50 |
| RTX PRO 6000 | 0.000842 | 3.03 |
| H100 | 0.001097 | 3.95 |
| H200 | 0.001261 | 4.54 |
| B200 / B300 | 0.001736 / 0.001972 | 6.25 / 7.10 |

- **CPU and memory:** CPU is $0.0000131 per physical core per second (about $0.047/h; a physical core is 2 vCPU), and memory is $0.00000222 per GiB per second. GPU containers pay these too.
- **Volumes:** $0.09/GiB-month, with 1 TiB/month included. Jobs are deleted after download, so storage is effectively free.
- **Egress:** $0.04/GiB. Downloading a 100–200 MB trajectory costs about $0.004–0.008 per seed.

oxDNA's own benchmarks show that time per step is nearly flat from 128 to 8,192 nucleotides, and an A100 is only about
1.4× a V100 there. For these design sizes (2k–14k nucleotides), the cheapest GPU (T4, or L4) is likely the best value
per simulation. Running several seeds on one GPU is the bigger lever. Confirm both with measurements.

## Checklist for the first real run

- [ ] The image builds. If the CMake step fails, check whether oxDNA v3.7.0 still builds with CUDA 12.8 and
      try `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- [ ] `scripts/modal_smoke.py` passes, and the bp fraction and RMSF match a local run of the same design and
      seed within seed noise.
- [ ] Record the oxDNA version on the local WSL build. It tracks master, while Modal is pinned to v3.7.0. Rerun the
      reference designs on Modal before mixing local and Modal grades in one comparison.
- [ ] Run `scripts/modal_gpu_benchmark.py` (below; it saves after each container, so rerunning resumes), then set the default GPU, `verlet_skin` and
      simulations per GPU from its `summary.md`.
- [ ] Check Modal's usage page after the smoke test, and compare it with the stage times in `log_*.txt`
      (`remote_seconds`).
- [ ] Decide whether to record Modal cost per run in `result.json`, next to the API cost.

## Benchmark: which GPU and settings are cheapest

`scripts/modal_gpu_benchmark.py` times production MD on Modal for three reference designs: the wireframe
tetrahedron (2,088 nt), the sharp triangle (7,900 nt) and the Rothemund rectangle (13,824 nt). It varies:

- the **GPU type** (default T4, L4, A10G; `--gpu CPU` adds a CPU run for small designs),
- **`verlet_skin`** (0.5 current, 0.2, 0.1; oxDNA's guide recommends 0.05–0.2),
- **simulations sharing one GPU** (1 and 3; `--mps` also tries NVIDIA MPS, if the container allows it).

```bash
uv run python scripts/modal_gpu_benchmark.py --dry-run     # matrix + worst-case cost (~$1.50 at defaults)
uv run python scripts/modal_gpu_benchmark.py --mps
```

How it works:
- Each design is relaxed once on Modal; the result is cached in `runs/_bench/modal-gpu/relax/`.
- Each (design, GPU) pair is one container that runs all of its variants, 30k MD steps each.
- The results go to `results.json`, `results.csv` and `summary.md`. For each variant they give ms/step,
  shape-protocol seeds per hour, and cost per grading seed (shape and standard protocols); for each design,
  the cheapest setup.
- Costs use oxDNA's own per-step timing, so container start-up is excluded.
- `E_pot` is a rough check that the `verlet_skin` variants simulate the same system. Before changing
  `verlet_skin` for grading, re-grade the reference designs.

## Benchmark results (2026-11-05, 30k MD steps per run, prices 2026-11-03)

oxDNA ms/step per simulation, from oxDNA's own timing. With 3 sims on one GPU this is each sim's speed
while sharing.

| design | T4 x1 | L4 x1 | A10G x1 | A100 x1 | L4 x3 (no MPS) | **L4 x3 + MPS** | A10G x3 + MPS |
|---|---|---|---|---|---|---|---|
| tetrahedron, 2,088 nt (skin 0.1) | 0.23 | 0.19 | 0.19 | 0.27 | 0.79 | **0.27** | 0.22 |
| triangle, 7,900 nt (skin 0.1) | 0.35 | 0.23 | 0.24 | 0.31 | 0.87 | **0.36** | 0.35 |
| Rothemund, 13,824 nt (skin 0.1) | 0.40 | 0.25 | 0.24 | 0.31 | 0.85 | **0.41** | 0.39 |

Cost of one `shape`-protocol grading seed (1.2M MD steps, MD only):

| design | current setup (L4 x1, skin 0.5) | L4 x3 + MPS, skin 0.1 |
|---|---|---|
| tetrahedron | $0.065 | **$0.028** |
| triangle | $0.081 | **$0.037** |
| Rothemund | $0.087 | **$0.042** |

Findings:
- **MPS works on Modal, and it is the main lever.** 3 sims sharing an L4 with MPS each run at about 0.6x
  single speed, which is 1.8x the throughput for the same GPU price.
- **Never share a GPU without MPS.** Plain time-slicing made each of 3 sims 3.5–4x slower, worse than
  running them one after another.
- **Bigger GPUs don't help at these sizes.** An A100 is no faster than an L4 (0.31 vs 0.25 ms/step on the
  Rothemund) at 2.6x the price; oxDNA is latency-bound below ~15k nt. The T4 is the slowest.
- **L4 is the cheapest at every size; A10G is close behind** and slightly faster with MPS.
- **verlet_skin 0.1 is 10–17% faster than 0.5.** Final potential energies agree within noise
  (±0.003 per nucleotide). Re-grade the references before adopting it.
- **L4 availability:** all L4 containers waited about 10–13 minutes for a GPU (839–974 s end to end against
  about 3 min of work); T4, A10G and A100 started at once. A10G is the natural fallback.
- Spend: $0.91 for this run (container time x price), plus relaxations.

## Open questions

- Is the per-stage design fast enough? Each stage pays for a container start and file transfer, which is
  small next to production MD, but noticeable for the `quick` protocol.
- Trajectory size: for very long runs, consider having oxDNA write fewer frames, or computing the mean structure
  remotely and returning only that.
