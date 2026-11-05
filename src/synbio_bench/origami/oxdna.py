"""Export scadnano designs to oxDNA and run a fixed relax -> production protocol.

oxDNA has no native Windows build, so the runner can call it through WSL
(`mode: wsl`), directly (`mode: local`, Linux/macOS or a native build), on a
Modal cloud GPU (`mode: modal`, see `modal_app.py`), or use a `mock` mode that
fakes a stable trajectory so the harness can be exercised without a simulator.
Mock results are always flagged in the returned metrics.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

import numpy as np

from origami_agent.design import load_design, nucleotide_map

from . import analysis, modal_runner

# One GPU, one simulation at a time, even when agents run in parallel threads. On Modal, up to
# `modal.max_parallel` groups run at once, each group being up to `modal.sims_per_gpu` simulations that share
# one GPU under MPS.
_SIM_SLOTS: dict[int, threading.BoundedSemaphore] = {}
_SIM_SLOTS_LOCK = threading.Lock()

# oxDNA2's FENE backbone diverges at r0 + delta = 0.7564 + 0.25 oxDNA units between backbone sites;
# production MD exits if any bond starts beyond it. Relaxation (with capped backbone forces) gets more
# rounds while a bond is still past this margin.
MAX_BACKBONE_BOND_OX = 0.98
EXTRA_RELAX_ROUNDS = 3


@dataclass
class OxDNAConfig:
    mode: str = "wsl"  # wsl | local | modal | mock
    executable: str = "oxDNA"
    wsl_distro: str | None = "Ubuntu"
    backend: str = "CUDA"  # CUDA | CPU; MC relaxation always runs on CPU
    interaction_type: str = "DNA2"
    temperature: str = "25C"
    salt_concentration: float = 1.0
    use_average_seq: bool = True
    seq_dep_file: str | None = None
    timeout_s: int = 3600
    # Neighbour-list margin (oxDNA length units, ~0.85 nm). Speed only: every interacting pair is always on the
    # list, but trajectories diverge numerically like a new seed. oxDNA suggests 0.05-0.2; 0.1 was 10-17% faster
    # than 0.5 in the Modal benchmark (docs/modal.md).
    verlet_skin: float = 0.5
    modal: dict[str, Any] = field(default_factory=dict)  # mode: modal settings; see modal_runner.DEFAULTS
    protocols: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "quick": {"mc_steps": 1000, "relax_steps": 2e5, "prod_steps": 5e5, "print_every": 1e4},
            "shape": {"mc_steps": 2000, "relax_steps": 2e5, "prod_steps": 1e6, "print_every": 2e4},
            "standard": {"mc_steps": 2000, "relax_steps": 1e6, "prod_steps": 5e6, "print_every": 5e4},
        }
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OxDNAConfig:
        d = dict(d)
        runner = d.pop("runner", {})
        return cls(**runner, **d)

    @property
    def max_parallel(self) -> int:
        """How many simulation groups may run at once with this runner."""
        return max(1, int(modal_runner.settings(self)["max_parallel"])) if self.mode == "modal" else 1

    @property
    def sims_per_gpu(self) -> int:
        """How many simulations share one GPU (run their stages together)."""
        return max(1, int(modal_runner.settings(self)["sims_per_gpu"])) if self.mode == "modal" else 1


def _sim_slot(cfg: OxDNAConfig) -> threading.BoundedSemaphore:
    n = cfg.max_parallel
    with _SIM_SLOTS_LOCK:
        return _SIM_SLOTS.setdefault(n, threading.BoundedSemaphore(n))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_design(design_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write initial.top/initial.dat, the designed-pair order-parameter file and an index map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    design = load_design(design_path)
    nm = nucleotide_map(design)
    dat, top = design.to_oxdna_format(warn_duplicate_strand_names=False)
    n_top = int(top.split(maxsplit=1)[0])
    if n_top != nm.n:
        raise RuntimeError(f"nucleotide map has {nm.n} particles but oxDNA export has {n_top}")

    (out / "initial.top").write_text(top)
    (out / "initial.dat").write_text(dat)
    pairs = nm.designed_pairs()
    op = ["{", "order_parameter = bond", "name = designed_pairs"]
    op += [f"pair{k} = {i}, {j}" for k, (i, j) in enumerate(pairs, start=1)]
    op.append("}")
    (out / "designed_pairs.op").write_text("\n".join(op) + "\n")
    (out / "index_map.json").write_text(
        json.dumps(
            {
                "slots": nm.slots,
                "strand_of": nm.strand_of,
                "scaffold_strands": [i for i, s in enumerate(design.strands) if s.is_scaffold],
                "pairs": pairs,
                "bases": "".join(nm.bases),
            }
        )
    )
    return {"n_nucleotides": nm.n, "n_strands": len(design.strands), "n_designed_pairs": len(pairs)}


# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------


def _common(cfg: OxDNAConfig, seed: int) -> dict[str, Any]:
    d: dict[str, Any] = {
        "interaction_type": cfg.interaction_type,
        "salt_concentration": cfg.salt_concentration,
        "T": cfg.temperature,
        "topology": "initial.top",
        "time_scale": "linear",
        "restart_step_counter": 1,
        "seed": seed,
        "verlet_skin": cfg.verlet_skin,
        "use_average_seq": int(cfg.use_average_seq),
        "max_io": 50,  # MB/s; oxDNA's 1 MB/s default aborts full-M13 designs (~14k nt) mid-run
    }
    if not cfg.use_average_seq and cfg.seq_dep_file:
        d["seq_dep_file"] = cfg.seq_dep_file
    return d


def build_inputs(cfg: OxDNAConfig, protocol: str, seed: int = 1) -> dict[str, dict[str, Any]]:
    p = cfg.protocols[protocol]
    gpu = cfg.backend.upper() == "CUDA"
    # max_density_multiplier: large sheets compact or roll during long relaxations and overflow
    # CUDA's auto-sized cells ("A cell contains more than _max_n_per_cell particles").
    md_backend = (
        {"backend": "CUDA", "backend_precision": "mixed", "max_density_multiplier": 10} if gpu else {"backend": "CPU"}
    )
    mc_steps, relax_steps, prod_steps = int(p["mc_steps"]), int(p["relax_steps"]), int(p["prod_steps"])
    every = int(p["print_every"])

    relax_mc = {
        **_common(cfg, seed),
        "backend": "CPU",
        "sim_type": "MC",
        "ensemble": "NVT",
        "delta_translation": 0.1,
        "delta_rotation": 0.1,
        "steps": mc_steps,
        "max_backbone_force": 5,
        "max_backbone_force_far": 10,
        "conf_file": "initial.dat",
        "lastconf_file": "relaxed_mc.dat",
        "trajectory_file": "traj_relax_mc.dat",
        "energy_file": "energy_relax_mc.dat",
        "print_conf_interval": mc_steps,
        "print_energy_every": max(1, mc_steps // 10),
    }
    relax_md = {
        **_common(cfg, seed),
        **md_backend,
        "sim_type": "MD",
        "dt": 0.002,
        "thermostat": "bussi",
        "bussi_tau": 1000,
        "newtonian_steps": 53,
        "diff_coeff": 1.0,
        "refresh_vel": 1,
        "steps": relax_steps,
        "max_backbone_force": 5,
        "max_backbone_force_far": 10,
        "conf_file": "relaxed_mc.dat",
        "lastconf_file": "relaxed.dat",
        "trajectory_file": "traj_relax_md.dat",
        "energy_file": "energy_relax_md.dat",
        "print_conf_interval": relax_steps,
        "print_energy_every": max(1, relax_steps // 20),
    }
    prod = {
        **_common(cfg, seed),
        **md_backend,
        "sim_type": "MD",
        "dt": 0.003,
        "thermostat": "john",
        "newtonian_steps": 103,
        "diff_coeff": 2.5,
        "refresh_vel": 1,
        "steps": prod_steps,
        "conf_file": "relaxed.dat",
        "lastconf_file": "last_conf.dat",
        "trajectory_file": "trajectory.dat",
        "energy_file": "energy.dat",
        "print_conf_interval": every,
        "print_energy_every": every,
        "data_output_1": (
            "{\n  name = hb_list.txt\n  print_every = %d\n"
            "  col_1 = {\n    type = hb_list\n    order_parameters_file = designed_pairs.op\n  }\n}" % every
        ),
    }
    return {"relax_mc": relax_mc, "relax_md": relax_md, "prod": prod}


def format_input(opts: dict[str, Any]) -> str:
    return "".join(f"{k} = {v}\n" for k, v in opts.items())


def write_input(path: Path, opts: dict[str, Any]) -> None:
    path.write_text(format_input(opts))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _to_wsl_path(p: Path) -> str:
    wp = PureWindowsPath(p.resolve())
    drive = wp.drive.rstrip(":").lower()
    return f"/mnt/{drive}/" + "/".join(wp.parts[1:])


def _run_stage(cfg: OxDNAConfig, work: Path, input_name: str, opts: dict[str, Any]) -> subprocess.CompletedProcess:
    if cfg.mode == "mock":
        return _mock_stage(work, opts)
    if cfg.mode == "wsl":
        cmd = ["wsl.exe"]
        if cfg.wsl_distro:
            cmd += ["-d", cfg.wsl_distro]
        cmd += ["--cd", _to_wsl_path(work), "--exec", cfg.executable, input_name]
    elif cfg.mode == "local":
        cmd = [cfg.executable, input_name]
    elif cfg.mode == "modal":
        return modal_runner.run_stage(cfg, work, input_name, opts)
    else:
        raise ValueError(f"unknown oxDNA runner mode {cfg.mode!r}")
    return subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=cfg.timeout_s)


def _mock_stage(work: Path, opts: dict[str, Any]) -> subprocess.CompletedProcess:
    """Fake a stable simulation: jitter the input config and report every designed pair bonded."""
    rng = np.random.default_rng(int(opts["seed"]))
    header, body = analysis.read_conf(work / opts["conf_file"])
    frames = []
    n_frames = max(1, int(opts["steps"]) // int(opts["print_conf_interval"]))
    for k in range(n_frames):
        b = body.copy()
        b[:, :3] += rng.normal(scale=0.05, size=(len(b), 3))
        frames.append(b)
        if k == n_frames - 1:
            analysis.write_conf(work / opts["lastconf_file"], header, b)
    with open(work / opts["trajectory_file"], "w") as f:
        for b in frames:
            f.write(analysis.format_conf(header, b))
    (work / opts["energy_file"]).write_text(f"{opts['steps']} -1.30 0.15 -1.15\n")
    if "data_output_1" in opts:
        pairs = json.loads((work / "index_map.json").read_text())["pairs"]
        with open(work / "hb_list.txt", "w") as f:
            for k in range(n_frames):
                f.write(f"# step {k}\n")
                f.writelines(f"{i} {j}\n" for i, j in pairs if rng.random() > 0.02)
    return subprocess.CompletedProcess(args=["mock"], returncode=0, stdout="mock oxDNA run\n", stderr="")


def _signal_note(returncode: int) -> str:
    """Negative exit codes are signals: -4 is SIGILL (binary built for a newer CPU), -9 SIGKILL (often OOM)."""
    if returncode >= 0:
        return ""
    try:
        return f" ({signal.Signals(-returncode).name})"
    except ValueError:
        return ""


def _tail(text: str, n: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


@dataclass
class SimJob:
    """One simulation of a design: where to write it, its seed, and what to score it against."""

    design_path: str | Path
    out_dir: str | Path
    seed: int = 1
    shape: dict[str, Any] | None = None
    wireframe: dict[str, Any] | None = None


def simulate(
    design_path: str | Path,
    out_dir: str | Path,
    cfg: OxDNAConfig,
    protocol: str = "quick",
    seed: int = 1,
    reanalyze: bool = False,
    shape: dict[str, Any] | None = None,
    wireframe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export, relax, run production MD and analyse. Never raises for simulation failures.

    With `reanalyze=True`, a previous successful run in `out_dir` is re-analysed
    instead of re-simulated (use after changing the analysis code).
    """
    return simulate_many([SimJob(design_path, out_dir, seed, shape, wireframe)], cfg, protocol, reanalyze)[0]


def design_size(design_path: str | Path) -> int:
    """Nucleotides in a design (0 if it can't be read); used to group similar sizes on one GPU."""
    try:
        return sum(strand.dna_length() for strand in load_design(design_path).strands)
    except Exception:  # noqa: BLE001 - export reports the real error later
        return 0


def simulate_many(jobs: list[SimJob], cfg: OxDNAConfig, protocol: str = "quick", reanalyze: bool = False,
                  on_result: Callable[[int, dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Simulate several jobs (seeds of one design, or of many designs); results come back in job order.

    Jobs are grouped `cfg.sims_per_gpu` at a time and a group moves through the stages together, so on Modal
    each stage of a group is one container whose simulations share the GPU under MPS. Jobs are sorted by design
    size first, so a group's simulations finish at about the same time instead of small ones waiting on a
    large one (a design's seeds stay together). Up to `cfg.max_parallel` groups run at once.
    Locally, groups are of one and run in the given order, one after another, as before.

    `on_result(job_index, result)` is called as each job's group finishes (from a worker thread on Modal),
    so callers can save results as they come.
    """
    k = cfg.sims_per_gpu
    order = list(range(len(jobs)))
    if k > 1 and len(jobs) > 1:
        sizes: dict[str, int] = {}
        for job in jobs:
            key = str(job.design_path)
            if key not in sizes:
                sizes[key] = design_size(job.design_path)
        order.sort(key=lambda i: (sizes[str(jobs[i].design_path)], i))
    groups = [order[i:i + k] for i in range(0, len(order), k)]
    results: list[dict[str, Any]] = [{} for _ in jobs]

    def run(group: list[int]) -> None:
        for i, r in zip(group, _simulate_group([jobs[i] for i in group], cfg, protocol, reanalyze)):
            results[i] = r
            if on_result:
                on_result(i, r)

    if len(groups) <= 1 or cfg.max_parallel <= 1:
        for g in groups:
            run(g)
    else:
        # Poll rather than block, so Ctrl+C reaches this thread on Windows too.
        pool = ThreadPoolExecutor(max_workers=min(len(groups), cfg.max_parallel))
        pending = {pool.submit(run, g) for g in groups}
        try:
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for f in done:
                    f.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
    return results


@dataclass
class _Sim:
    job: SimJob
    work: Path
    result: dict[str, Any]
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    alive: bool = True


def _run_stages(cfg: OxDNAConfig, items: list[tuple[Path, str, dict[str, Any]]]) -> list[Any]:
    """Per item, a CompletedProcess or the exception (timeout, runner not found) for that item."""
    if cfg.mode == "modal":
        return modal_runner.run_stages(cfg, items)
    out: list[Any] = []
    for work, input_name, opts in items:
        try:
            out.append(_run_stage(cfg, work, input_name, opts))
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            out.append(e)
    return out


def _label(job: SimJob) -> str:
    """Short name for progress lines: task/model/rep#seed for benchmark runs, else the work dir's name."""
    parts = Path(job.out_dir).parts
    if "eval" in parts:
        i = len(parts) - 1 - parts[::-1].index("eval")
        return "/".join(parts[max(0, i - 3):i]) + f"#{job.seed}"
    return f"{Path(job.out_dir).name}#{job.seed}"


def _stage(cfg: OxDNAConfig, sims: list[_Sim], stage: str, opts_of: dict[int, dict[str, Any]]) -> None:
    """Run `stage` for the given sims together; a sim that fails is marked dead with the error recorded."""
    items = []
    for sim in sims:
        write_input(sim.work / f"input_{stage}", opts_of[id(sim)])
        items.append((sim.work, f"input_{stage}", opts_of[id(sim)]))
    labels = ", ".join(_label(sim.job) for sim in sims)
    if cfg.mode != "mock":
        print(f"[sim ] {stage:<10} start  {len(sims)} sim(s): {labels}", flush=True)
    t0 = time.time()
    results = _run_stages(cfg, items)
    seconds = round(time.time() - t0, 1)
    if cfg.mode != "mock":
        print(f"[sim ] {stage:<10} done   {seconds:>6.0f} s: {labels}", flush=True)
    for sim, proc in zip(sims, results):
        opts = opts_of[id(sim)]
        if isinstance(proc, subprocess.TimeoutExpired):
            sim.result.update(ok=False, failed_stage=stage, error=f"timed out after {cfg.timeout_s}s")
            sim.alive = False
            continue
        if isinstance(proc, FileNotFoundError):
            sim.result.update(ok=False, failed_stage=stage, error=f"oxDNA runner not found: {proc}")
            sim.alive = False
            continue
        if isinstance(proc, BaseException):
            raise proc
        sim.timings[stage] = seconds
        (sim.work / f"log_{stage}.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0 or not (sim.work / opts["lastconf_file"]).exists():
            sim.result.update(
                ok=False,
                failed_stage=stage,
                error=f"oxDNA exited with code {proc.returncode}{_signal_note(proc.returncode)}",
                log_tail=_tail(proc.stderr or proc.stdout),
            )
            sim.alive = False


def _simulate_group(jobs: list[SimJob], cfg: OxDNAConfig, protocol: str, reanalyze: bool) -> list[dict[str, Any]]:
    sims: list[_Sim] = []
    for job in jobs:
        work = Path(job.out_dir)
        previous = work / "result.json"
        if reanalyze and previous.exists() and json.loads(previous.read_text()).get("ok"):
            result = json.loads(previous.read_text())
            result["metrics"] = analysis.analyze_run(work, job.shape, job.wireframe)
            previous.write_text(json.dumps(result, indent=2))
            sims.append(_Sim(job, work, result, alive=False))
            continue
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        result: dict[str, Any] = {"protocol": protocol, "seed": job.seed, "runner": cfg.mode,
                                  "verlet_skin": cfg.verlet_skin, "work_dir": str(work)}
        if cfg.mode == "mock":
            result["WARNING"] = "mock oxDNA runner: metrics are synthetic and do not reflect real physics"
        sim = _Sim(job, work, result)
        sims.append(sim)
        try:
            result["export"] = export_design(job.design_path, work)
        except Exception as e:  # noqa: BLE001
            result.update(ok=False, failed_stage="export", error=f"{type(e).__name__}: {e}")
            sim.alive = False
            continue
        result["initial_backbone"] = analysis.backbone_report(work / "initial.top", work / "initial.dat",
                                                              work / "index_map.json")
        sim.inputs = build_inputs(cfg, protocol, job.seed)
    simulated = [sim for sim in sims if sim.alive]
    if len(simulated) > 1:
        for sim in simulated:
            sim.result["shared_gpu_with"] = len(simulated) - 1

    with _sim_slot(cfg):
        for stage in ("relax_mc", "relax_md", "prod"):
            alive = [sim for sim in sims if sim.alive]
            if not alive:
                break
            _stage(cfg, alive, stage, {id(sim): sim.inputs[stage] for sim in alive})
            if stage == "relax_md" and cfg.mode != "mock":
                # Production MD refuses to start if any bond is still longer than the backbone potential
                # allows (e.g. a long unpaired loop still being pulled in). Keep relaxing for a few rounds.
                for k in range(1, EXTRA_RELAX_ROUNDS + 1):
                    need = []
                    for sim in (s_ for s_ in sims if s_.alive):
                        worst = analysis.max_backbone_site_bond(sim.work / "initial.top", sim.work / "relaxed.dat")
                        sim.result["relaxed_max_backbone_bond"] = round(worst, 3)
                        if worst > MAX_BACKBONE_BOND_OX:
                            need.append(sim)
                    if not need:
                        break
                    more = {id(sim): {**sim.inputs["relax_md"], "conf_file": "relaxed.dat",
                                      "lastconf_file": "relaxed.dat", "trajectory_file": f"traj_relax_md_{k}.dat",
                                      "energy_file": f"energy_relax_md_{k}.dat"} for sim in need}
                    _stage(cfg, need, f"relax_md_{k}", more)
                    for sim in need:
                        if sim.alive:
                            sim.result["extra_relax_rounds"] = k

    for sim in sims:
        if not sim.alive:
            continue
        sim.result["stage_seconds"] = sim.timings
        try:
            sim.result["metrics"] = analysis.analyze_run(sim.work, sim.job.shape, sim.job.wireframe)
            sim.result["ok"] = True
        except Exception as e:  # noqa: BLE001
            sim.result.update(ok=False, failed_stage="analysis", error=f"{type(e).__name__}: {e}")
        (sim.work / "result.json").write_text(json.dumps(sim.result, indent=2))
    return [sim.result for sim in sims]


def config_from_env(cfg: OxDNAConfig) -> OxDNAConfig:
    """Let OXDNA_MODE / OXDNA_EXECUTABLE / OXDNA_BACKEND / OXDNA_MODAL_GPU / OXDNA_VERLET_SKIN override the config."""
    if m := os.environ.get("OXDNA_MODE"):
        cfg.mode = m
    if exe := os.environ.get("OXDNA_EXECUTABLE"):
        cfg.executable = exe
    if b := os.environ.get("OXDNA_BACKEND"):
        cfg.backend = b
    if v := os.environ.get("OXDNA_VERLET_SKIN"):
        cfg.verlet_skin = float(v)
    if g := os.environ.get("OXDNA_MODAL_GPU"):
        cfg.modal = {**cfg.modal, "gpu": g}
    return cfg

