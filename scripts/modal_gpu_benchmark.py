"""Benchmark oxDNA speed and cost per graded design on Modal GPUs.

Matrix: design size (2,088 / 7,900 / 13,824 nucleotides) x GPU type x verlet_skin x simulations per GPU (x MPS).
Each design is relaxed once on Modal (cached under --out), then every (design, GPU) pair is one container
call that runs all its variants back to back on that design's relaxed structure, using production MD settings
(the `shape` protocol). Nothing but timings comes back.

Prereqs: the app is deployed (docs/modal.md). Check the matrix and a cost ceiling first:

    uv run python scripts/modal_gpu_benchmark.py --dry-run
    uv run python scripts/modal_gpu_benchmark.py                      # default matrix
    uv run python scripts/modal_gpu_benchmark.py --gpu T4 --design rothemund --skin 0.5 --skin 0.2 --mps

Writes results.json (raw), results.csv and summary.md to --out (default runs/_bench/modal-gpu).

Cost columns use oxDNA's own per-step time, so container start-up and oxDNA initialisation are excluded,
because they are amortised over a real grading run. "$/seed" is the MD part of one grading seed:
shape = 0.2M relax + 1M production steps; standard = 1M + 5M. verlet_skin only changes speed, not the
model, but trajectories diverge numerically. Re-grade the reference designs before changing it for real.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import statistics
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import yaml

from synbio_bench.origami import modal_runner as mr
from synbio_bench.origami.oxdna import OxDNAConfig, build_inputs, config_from_env, format_input, simulate

ROOT = Path(__file__).resolve().parents[1]

DESIGNS = {  # name: (reference script, extra args, nucleotides incl. staples)
    "tetrahedron": ("reference_designs/wireframe_tetrahedron.py", [], 2088),  # 1,044-nt scaffold
    "triangle": ("reference_designs/sharp_triangle.py", [], 7900),  # ~4,000-nt scaffold
    "rothemund": ("reference_designs/rothemund.py", ["rect"], 13824),
}
RELAX_PROTOCOL = {"mc_steps": 2000, "relax_steps": 2e5, "prod_steps": 1e4, "print_every": 1e4}
MD_STEPS_PER_SEED = {"shape": 1.2e6, "standard": 6.0e6}
GPU_CPU_CORES, CPU_CPU_CORES, MEM_GIB = 2.0, 4.0, 4.0  # match bench_gpu / bench_cpu in modal_app.py


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_modal_runner.py)
# ---------------------------------------------------------------------------


def price_per_s(gpu: str) -> float:
    """Container price per second: GPU plus the CPU and memory the bench function reserves."""
    if gpu == "CPU":
        return CPU_CPU_CORES * mr.CPU_PRICE_PER_CORE_S + MEM_GIB * mr.MEM_PRICE_PER_GIB_S
    if gpu not in mr.GPU_PRICE_PER_S:
        raise ValueError(f"no price for GPU {gpu!r}; known: {', '.join(mr.GPU_PRICE_PER_S)}")
    return mr.GPU_PRICE_PER_S[gpu] + GPU_CPU_CORES * mr.CPU_PRICE_PER_CORE_S + MEM_GIB * mr.MEM_PRICE_PER_GIB_S


def timed_opts(cfg: OxDNAConfig, steps: int, skin: float, seed: int, cpu: bool) -> dict[str, Any]:
    """Production MD input (shape protocol output cadence) starting from the relaxed structure."""
    opts = {**build_inputs(cfg, "shape", seed)["prod"], "steps": int(steps), "verlet_skin": skin,
            "conf_file": "relaxed.dat", "seed": seed}
    if cpu:
        for k in ("backend_precision", "max_density_multiplier"):
            opts.pop(k, None)
        opts["backend"] = "CPU"
    return opts


def variants_for(cfg: OxDNAConfig, gpu: str, steps: int, skins: list[float], copies: list[int],
                 mps: bool) -> list[dict[str, Any]]:
    cpu = gpu == "CPU"
    out = []
    for skin in skins:
        for n in copies:
            for use_mps in [False] + ([True] if mps and n > 1 and not cpu else []):
                out.append({
                    "name": f"skin{skin}-x{n}{'-mps' if use_mps else ''}",
                    "skin": skin,
                    "mps": use_mps,
                    "inputs": [format_input(timed_opts(cfg, steps, skin, seed=1 + k, cpu=cpu)) for k in range(n)],
                })
    return out


def rows_from(design: str, n_nt: int, gpu: str, steps: int, result: dict[str, Any],
              variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skins = {v["name"]: v["skin"] for v in variants}  # names look like "skin0.2-x3-mps"
    for v in result["variants"]:
        skins.setdefault(v["name"], float(v["name"].split("-")[0].removeprefix("skin")))
    rows = []
    for v in result["variants"]:
        row: dict[str, Any] = {"design": design, "n_nt": n_nt, "gpu": gpu, "gpu_name": result.get("gpu"),
                               "skin": skins.get(v["name"]), "copies": v["copies"], "mps": v["mps"],
                               "steps": steps}
        if v.get("skipped"):
            rows.append({**row, "ok": False, "note": v["skipped"]})
            continue
        runs = v["runs"]
        per_step = [r["per_step_ms"] for r in runs if r.get("per_step_ms")]
        ok = all(r["returncode"] == 0 and not r["timed_out"] for r in runs) and len(per_step) == len(runs)
        row.update(ok=ok, wall_s=v["wall_s"],
                   note=None if ok else next((r.get("log_tail") for r in runs if r.get("log_tail")), "failed"))
        if per_step:
            ms = statistics.mean(per_step)
            # Each process's per-step time already includes slow-down from sharing the GPU.
            usd_per_mstep = price_per_s(gpu) * (ms / 1000) * 1e6 / v["copies"]
            energies = []
            for r in runs:
                try:
                    energies.append(float(r["last_energy"].split()[1]))
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
            row.update(
                per_step_ms=round(ms, 4),
                sims_per_hour_shape=round(3600 / (ms / 1000 * MD_STEPS_PER_SEED["shape"]) * v["copies"], 2),
                usd_per_million_steps=round(usd_per_mstep, 4),
                usd_per_seed_shape=round(usd_per_mstep * MD_STEPS_PER_SEED["shape"] / 1e6, 4),
                usd_per_seed_standard=round(usd_per_mstep * MD_STEPS_PER_SEED["standard"] / 1e6, 4),
                e_pot_mean=round(statistics.mean(energies), 5) if energies else None,
            )
        rows.append(row)
    return rows


def ceiling_usd(n_calls_by_gpu: dict[str, int], variants_by_gpu: dict[str, list[dict]], steps: int,
                assume_ms: float, n_relax: int) -> float:
    """Rough upper bound: no speed-up from sharing a GPU, 60 s start-up per call, T4 relaxations."""
    total = 0.0
    for gpu, n_calls in n_calls_by_gpu.items():
        ms = assume_ms * (10 if gpu == "CPU" else 1)
        seconds = sum(len(v["inputs"]) * steps * ms / 1000 for v in variants_by_gpu[gpu]) + 60
        total += n_calls * seconds * price_per_s(gpu)
    total += n_relax * (2e5 * assume_ms / 1000 + 120) * price_per_s("T4")
    return total


def summary_md(rows: list[dict[str, Any]], spent: float | None) -> str:
    lines = [f"# oxDNA on Modal: speed and cost per seed (prices {mr.PRICES_AS_OF})", ""]
    if spent is not None:
        lines += [f"Benchmark spend (container time x price, excl. relaxation): ${spent:.2f}", ""]
    for design in dict.fromkeys(r["design"] for r in rows):
        rs = [r for r in rows if r["design"] == design]
        lines += [f"## {design} ({rs[0]['n_nt']} nt)", "",
                  "| GPU | skin | sims/GPU | MPS | ms/step | shape seeds/h | $/seed shape | $/seed standard | E_pot | note |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: r.get("usd_per_seed_shape") or float("inf")):
            note = (r.get("note") or "").splitlines()[0][:60] if r.get("note") else ""
            lines.append(f"| {r['gpu']} | {r.get('skin', '–')} | {r.get('copies', '–')} | {'yes' if r.get('mps') else ''} | "
                         f"{r.get('per_step_ms', '–')} | {r.get('sims_per_hour_shape', '–')} | "
                         f"{r.get('usd_per_seed_shape', '–')} | {r.get('usd_per_seed_standard', '–')} | "
                         f"{r.get('e_pot_mean', '–')} | {note} |")
        best = min((r for r in rs if r.get("ok")), key=lambda r: r["usd_per_seed_shape"], default=None)
        if best:
            lines += ["", f"Cheapest: **{best['gpu']}**, skin {best['skin']}, {best['copies']} per GPU"
                          f"{' with MPS' if best['mps'] else ''}: ${best['usd_per_seed_shape']} per shape seed."]
        lines.append("")
    lines += ["E_pot is the mean final potential energy per nucleotide across copies: a sanity check that",
              "verlet_skin variants simulate the same system, not a convergence test."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Modal calls
# ---------------------------------------------------------------------------


def build_design(name: str, out: Path) -> Path:
    script, extra, _ = DESIGNS[name]
    path = out / "designs" / f"{name}.sc"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, Path(script).name, *extra, str(path)], cwd=ROOT / Path(script).parent,
                       check=True, capture_output=True)
    return path


def relax(name: str, cfg: OxDNAConfig, out: Path) -> tuple[Path, int]:
    """Relax once on Modal (cached). Returns the work dir holding relaxed.dat and the nucleotide count."""
    work = out / "relax" / name
    done = work / "result.json"
    if done.exists() and json.loads(done.read_text()).get("ok") and (work / "relaxed.dat").exists():
        return work, json.loads(done.read_text())["export"]["n_nucleotides"]
    cfg.protocols = {**cfg.protocols, "bench_relax": RELAX_PROTOCOL}
    print(f"[relax] {name} on Modal ...", flush=True)
    r = simulate(build_design(name, out), work, cfg, protocol="bench_relax", seed=1)
    if not r.get("ok"):
        raise SystemExit(f"relaxation of {name} failed at {r.get('failed_stage')}: {r.get('error')}\n{r.get('log_tail')}")
    return work, r["export"]["n_nucleotides"]


# Modal calls in flight, so Ctrl+C can cancel them (key: "design on gpu").
ACTIVE_CALLS: dict[str, Any] = {}
_ACTIVE_LOCK = threading.Lock()


class NoResult(RuntimeError):
    """A container call gave no result in time (usually: no GPU of that type was free) and was cancelled."""


def call_bench(gpu: str, files: dict[str, bytes], variants: list[dict], timeout_s: int,
               environment: str | None, wait_s: float) -> dict:
    """Run one container's variants. Gives up (and cancels the call) after wait_s, including GPU queueing."""
    modal = mr._modal()
    if gpu == "CPU":
        fn = modal.Function.from_name(mr.APP_NAME, mr.BENCH_CPU_FUNCTION, environment_name=environment)
    else:
        fn = modal.Function.from_name(mr.APP_NAME, mr.BENCH_GPU_FUNCTION, environment_name=environment)
        fn = fn.with_options(gpu=gpu)
    payload = [{k: v[k] for k in ("name", "inputs", "mps")} for v in variants]
    call = fn.spawn(files, payload, timeout_s)
    key = f"{id(call)}"
    with _ACTIVE_LOCK:
        ACTIVE_CALLS[key] = call
    try:
        return call.get(timeout=wait_s)
    except Exception as e:  # noqa: BLE001
        if isinstance(e, TimeoutError) or mr._is_timeout(e):
            try:
                call.cancel()
            except Exception:  # noqa: BLE001
                pass
            raise NoResult(f"no result after {wait_s / 60:.0f} min (no {gpu} free?); call cancelled") from e
        raise
    finally:
        with _ACTIVE_LOCK:
            ACTIVE_CALLS.pop(key, None)


def cancel_active() -> int:
    with _ACTIVE_LOCK:
        calls = list(ACTIVE_CALLS.values())
    for call in calls:
        try:
            call.cancel()
        except Exception:  # noqa: BLE001
            pass
    return len(calls)


def load_previous(out: Path, steps: int) -> list[dict]:
    """Successful (design, GPU) results from an earlier run with the same step count."""
    path = out / "results.json"
    if not path.exists():
        return []
    prev = json.loads(path.read_text())
    if prev.get("steps") != steps:
        print(f"[note] ignoring {path}: it used {prev.get('steps')} steps, this run uses {steps}")
        return []
    return [r for r in prev.get("raw", []) if "result" in r]


def rows_for(raw: list[dict]) -> list[dict]:
    rows = []
    for r in raw:
        if "result" in r:
            rows += rows_from(r["design"], r["n_nt"], r["gpu"], r["steps"], r["result"], [])
        else:
            rows.append({"design": r["design"], "n_nt": r["n_nt"], "gpu": r["gpu"], "ok": False, "note": r["error"]})
    return rows


def save(out: Path, steps: int, raw: list[dict]) -> str:
    rows = rows_for(raw)
    spent = sum(r["result"]["container_s"] * price_per_s(r["gpu"]) for r in raw if "result" in r)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"prices_as_of": mr.PRICES_AS_OF, "steps": steps,
                                                  "raw": raw, "rows": rows}, indent=1))
    with open(out / "results.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = summary_md(rows, spent)
    (out / "summary.md").write_text(md)
    return md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", action="append", choices=list(DESIGNS), help="default: all three")
    p.add_argument("--gpu", action="append", help="Modal GPU type (repeatable), or CPU; default: T4, L4, A10G")
    p.add_argument("--skin", action="append", type=float, help="verlet_skin values; default: 0.5, 0.2, 0.1")
    p.add_argument("--copies", action="append", type=int, help="simulations per GPU; default: 1 and 3")
    p.add_argument("--mps", action="store_true", help="also try NVIDIA MPS for copies > 1")
    p.add_argument("--steps", type=int, default=30_000, help="MD steps per timed run (CPU runs use a tenth)")
    p.add_argument("--cpu-max-nt", type=int, default=5000, help="skip CPU runs for designs larger than this")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per variant")
    p.add_argument("--wait", type=float, default=30, help="minutes to wait for one container's result, including "
                                                          "time queued for a GPU; then it is cancelled")
    p.add_argument("--parallel", type=int, default=3, help="containers at once")
    p.add_argument("--fresh", action="store_true", help="ignore earlier results in --out and rerun everything")
    p.add_argument("--assume-ms", type=float, default=1.5, help="ms/step assumed for the --dry-run cost ceiling")
    p.add_argument("--out", default=str(ROOT / "runs" / "_bench" / "modal-gpu"))
    p.add_argument("--dry-run", action="store_true", help="print the matrix and a cost ceiling; call nothing")
    args = p.parse_args()

    designs = args.design or list(DESIGNS)
    gpus = args.gpu or ["T4", "L4", "A10G"]
    skins = args.skin or [0.5, 0.2, 0.1]
    copies = sorted(set(args.copies or [1, 3]))
    out = Path(args.out)
    for g in gpus:
        price_per_s(g)  # fail early on an unknown GPU name

    cfg = config_from_env(OxDNAConfig.from_dict(yaml.safe_load((ROOT / "configs" / "oxdna.yaml").read_text())))
    cfg.mode = "modal"

    def steps_for(gpu: str) -> int:
        return args.steps // 10 if gpu == "CPU" else args.steps

    pairs = [(d, g) for d in designs for g in gpus if not (g == "CPU" and DESIGNS[d][2] > args.cpu_max_nt)]
    raw = [] if args.fresh else load_previous(out, args.steps)
    done = {(r["design"], r["gpu"]) for r in raw}
    if done & set(pairs):
        print(f"[resume] {len(done & set(pairs))} design/GPU pairs already done in {out} "
              f"(--fresh to rerun): {', '.join(f'{d} on {g}' for d, g in pairs if (d, g) in done)}")
    pairs = [pr for pr in pairs if pr not in done]
    variants_by_gpu = {g: variants_for(cfg, g, steps_for(g), skins, copies, args.mps) for g in gpus}
    print(f"{len(pairs)} container calls to make, {len(variants_by_gpu[gpus[0]])} variants each: "
          + ", ".join(v["name"] for v in variants_by_gpu[gpus[0]]))
    calls_by_gpu = {g: sum(1 for _, gg in pairs if gg == g) for g in gpus}
    print(f"cost ceiling ~${ceiling_usd(calls_by_gpu, variants_by_gpu, args.steps, args.assume_ms, len(designs)):.2f} "
          f"(assumes {args.assume_ms} ms/step, no gain from sharing a GPU; prices {mr.PRICES_AS_OF})")
    if args.dry_run or not pairs:
        if not args.dry_run and raw:
            print(save(out, args.steps, raw))
        return

    relaxed = {}
    for d in dict.fromkeys(d for d, _ in pairs):
        try:
            relaxed[d] = relax(d, cfg, out)
        except mr.ModalUnavailable as e:
            raise SystemExit(f"Modal is not ready: {e}") from e

    def run_pair(pair: tuple[str, str]) -> dict:
        design, gpu = pair
        work, n_nt = relaxed[design]
        entry: dict[str, Any] = {"design": design, "gpu": gpu, "n_nt": n_nt, "steps": steps_for(gpu)}
        names = [n for n in mr.stage_inputs(timed_opts(cfg, 1, 0.5, 1, gpu == "CPU"), "input") if n != "input"]
        files = {n: (work / n).read_bytes() for n in names}
        print(f"[bench] {design} on {gpu} ...", flush=True)
        t0 = time.time()
        try:
            res = call_bench(gpu, files, variants_by_gpu[gpu], args.timeout, mr.settings(cfg)["environment"],
                             args.wait * 60)
        except Exception as e:  # noqa: BLE001 - one bad container must not lose the others' results
            print(f"[fail ] {design} on {gpu}: {type(e).__name__}: {e}", flush=True)
            return {**entry, "error": f"{type(e).__name__}: {e}"}
        print(f"[bench] {design} on {gpu} ({res.get('gpu')}): {round(time.time() - t0)} s", flush=True)
        return {**entry, "result": res}

    # Results are saved after every container, so stopping early keeps what has finished. The main thread
    # polls (rather than blocking on the pool) so Ctrl+C works on Windows too; it cancels the Modal calls
    # still running and exits without waiting for their threads.
    pool = ThreadPoolExecutor(max_workers=args.parallel)
    pending = {pool.submit(run_pair, pr) for pr in pairs}
    try:
        while pending:
            finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for f in finished:
                raw.append(f.result())
                save(out, args.steps, raw)
    except KeyboardInterrupt:
        n = cancel_active()
        if raw:
            save(out, args.steps, raw)
        print(f"\n[stop ] cancelled {n} Modal call(s); {len(raw)} finished container(s) saved in {out}. "
              f"Rerun the same command to resume.", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    pool.shutdown()
    print(save(out, args.steps, raw))

if __name__ == "__main__":
    main()
