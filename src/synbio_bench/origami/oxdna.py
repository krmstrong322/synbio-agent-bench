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
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from origami_agent.design import load_design, nucleotide_map

from . import analysis, modal_runner

# One GPU, one simulation at a time, even when agents run in parallel threads. Modal runs several at once
# (each on its own GPU), up to `modal.max_parallel`.
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
        """How many simulations may run at once with this runner."""
        return max(1, int(modal_runner.settings(self)["max_parallel"])) if self.mode == "modal" else 1


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
        "verlet_skin": 0.5,
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


def _tail(text: str, n: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


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
    work = Path(out_dir)
    previous = work / "result.json"
    if reanalyze and previous.exists() and json.loads(previous.read_text()).get("ok"):
        result = json.loads(previous.read_text())
        result["metrics"] = analysis.analyze_run(work, shape, wireframe)
        previous.write_text(json.dumps(result, indent=2))
        return result
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result: dict[str, Any] = {"protocol": protocol, "seed": seed, "runner": cfg.mode, "work_dir": str(work)}
    if cfg.mode == "mock":
        result["WARNING"] = "mock oxDNA runner: metrics are synthetic and do not reflect real physics"

    try:
        result["export"] = export_design(design_path, work)
    except Exception as e:  # noqa: BLE001
        result.update(ok=False, failed_stage="export", error=f"{type(e).__name__}: {e}")
        return result

    result["initial_backbone"] = analysis.backbone_report(work / "initial.top", work / "initial.dat", work / "index_map.json")
    inputs = build_inputs(cfg, protocol, seed)
    timings: dict[str, float] = {}

    def run(stage: str, opts: dict[str, Any]) -> bool:
        write_input(work / f"input_{stage}", opts)
        t0 = time.time()
        try:
            proc = _run_stage(cfg, work, f"input_{stage}", opts)
        except subprocess.TimeoutExpired:
            result.update(ok=False, failed_stage=stage, error=f"timed out after {cfg.timeout_s}s")
            return False
        except FileNotFoundError as e:
            result.update(ok=False, failed_stage=stage, error=f"oxDNA runner not found: {e}")
            return False
        timings[stage] = round(time.time() - t0, 1)
        (work / f"log_{stage}.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0 or not (work / opts["lastconf_file"]).exists():
            result.update(
                ok=False,
                failed_stage=stage,
                error=f"oxDNA exited with code {proc.returncode}",
                log_tail=_tail(proc.stderr or proc.stdout),
            )
            return False
        return True

    with _sim_slot(cfg):
        for stage, opts in inputs.items():
            if not run(stage, opts):
                return result
            if stage == "relax_md" and cfg.mode != "mock":
                # Production MD refuses to start if any bond is still longer than the backbone potential
                # allows (e.g. a long unpaired loop still being pulled in). Keep relaxing for a few rounds.
                for k in range(1, EXTRA_RELAX_ROUNDS + 1):
                    worst = analysis.max_backbone_site_bond(work / "initial.top", work / "relaxed.dat")
                    result["relaxed_max_backbone_bond"] = round(worst, 3)
                    if worst <= MAX_BACKBONE_BOND_OX:
                        break
                    more = {**opts, "conf_file": "relaxed.dat", "lastconf_file": "relaxed.dat",
                            "trajectory_file": f"traj_relax_md_{k}.dat", "energy_file": f"energy_relax_md_{k}.dat"}
                    if not run(f"relax_md_{k}", more):
                        return result
                    result["extra_relax_rounds"] = k
    result["stage_seconds"] = timings

    try:
        result["metrics"] = analysis.analyze_run(work, shape, wireframe)
        result["ok"] = True
    except Exception as e:  # noqa: BLE001
        result.update(ok=False, failed_stage="analysis", error=f"{type(e).__name__}: {e}")
    (work / "result.json").write_text(json.dumps(result, indent=2))
    return result


def config_from_env(cfg: OxDNAConfig) -> OxDNAConfig:
    """Let OXDNA_MODE / OXDNA_EXECUTABLE / OXDNA_BACKEND / OXDNA_MODAL_GPU override the config file."""
    if m := os.environ.get("OXDNA_MODE"):
        cfg.mode = m
    if exe := os.environ.get("OXDNA_EXECUTABLE"):
        cfg.executable = exe
    if b := os.environ.get("OXDNA_BACKEND"):
        cfg.backend = b
    if g := os.environ.get("OXDNA_MODAL_GPU"):
        cfg.modal = {**cfg.modal, "gpu": g}
    return cfg

