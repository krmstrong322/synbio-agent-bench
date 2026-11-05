"""Client for `mode: modal`: run one oxDNA stage on Modal and bring its outputs back.

The deployed app is defined in `modal_app.py`. This side ships the files a stage reads, calls the deployed
function (GPU for CUDA stages, CPU for MC relaxation), downloads what the stage wrote from the Modal Volume
into the local work dir, and returns a `CompletedProcess` so `oxdna.simulate()` treats it like any other
runner. Relax rounds, backbone checks and all analysis stay local.

`modal` is an optional dependency (`uv sync --extra modal`) and is imported only when a stage runs.
"""

from __future__ import annotations

import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

# Must match modal_app.py (tests/test_modal_runner.py checks this without importing modal).
APP_NAME = "synbio-oxdna"
VOLUME_NAME = "synbio-oxdna-runs"
JOBS_DIR = "jobs"
GPU_FUNCTION = "run_stages_gpu"
CPU_FUNCTION = "run_stages_cpu"
BENCH_GPU_FUNCTION = "bench_gpu"
BENCH_CPU_FUNCTION = "bench_cpu"

# modal.com/pricing, read 2026-11-03. USD per second (GPU), per physical core-second (CPU), per GiB-second.
PRICES_AS_OF = "2026-11-03"
GPU_PRICE_PER_S = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "A10G": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100": 0.000583, "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842,
    "H100": 0.001097, "H200": 0.001261, "B200": 0.001736, "B300": 0.001972,
}
CPU_PRICE_PER_CORE_S = 0.0000131
MEM_PRICE_PER_GIB_S = 0.00000222

DEFAULTS: dict[str, Any] = {
    "app": APP_NAME,
    "volume": VOLUME_NAME,
    "gpu": None,  # None: the GPUs the app was deployed with (L4, else A10G; OXDNA_MODAL_GPU at deploy)
    "sims_per_gpu": 3,  # simulations sharing one GPU under MPS (seeds of a design); 1 = one GPU each
    "max_parallel": 4,  # groups (containers) at once; Modal's Starter plan allows 10 GPUs at once
    "environment": None,  # Modal environment name; None uses the workspace default
}

# Input-file options a stage may read from its working directory.
_INPUT_KEYS = ("topology", "conf_file", "seq_dep_file", "external_forces_file")
_OP_FILE = re.compile(r"order_parameters_file\s*=\s*(\S+)")


class ModalUnavailable(RuntimeError):
    """Modal is not installed, not logged in, or the app is not deployed. Not a simulation failure."""


def settings(cfg: Any) -> dict[str, Any]:
    return {**DEFAULTS, **(getattr(cfg, "modal", None) or {})}


def stage_inputs(opts: dict[str, Any], input_name: str) -> list[str]:
    """Files a stage needs from its work dir: the input file, topology, starting conf and any referenced files."""
    names = [input_name]
    for key in _INPUT_KEYS:
        if opts.get(key):
            names.append(str(opts[key]))
    for value in opts.values():
        if isinstance(value, str):
            names += _OP_FILE.findall(value)
    out: list[str] = []
    for n in names:
        if Path(n).is_absolute() or "/" in n or "\\" in n:
            raise ValueError(f"modal runner needs stage files inside the work dir, got {n!r}")
        if n not in out:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Modal handles (patched in tests)
# ---------------------------------------------------------------------------


def _modal():
    try:
        import modal
    except ImportError as e:
        raise ModalUnavailable("mode: modal needs the modal package: `uv sync --extra modal`") from e
    return modal


# Modal calls in flight, so an interrupted evaluation can cancel them (they would otherwise keep running, billed).
_ACTIVE: dict[int, Any] = {}
_ACTIVE_LOCK = threading.Lock()


def cancel_all() -> int:
    """Cancel every Modal call this process still has running. Returns how many were cancelled."""
    with _ACTIVE_LOCK:
        calls = list(_ACTIVE.values())
        _ACTIVE.clear()
    for call in calls:
        try:
            call.cancel()
        except Exception:  # noqa: BLE001
            pass
    return len(calls)


def _function(s: dict[str, Any], gpu: bool) -> Callable[..., dict]:
    modal = _modal()
    fn = modal.Function.from_name(s["app"], GPU_FUNCTION if gpu else CPU_FUNCTION,
                                  environment_name=s["environment"])
    if gpu and s["gpu"]:
        fn = fn.with_options(gpu=s["gpu"])

    def call(*args: Any) -> dict:
        fc = fn.spawn(*args)
        with _ACTIVE_LOCK:
            _ACTIVE[id(fc)] = fc
        try:
            return fc.get()
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.pop(id(fc), None)

    return call


def _volume(s: dict[str, Any]) -> Any:
    return _modal().Volume.from_name(s["volume"], environment_name=s["environment"])


def _is_timeout(e: Exception) -> bool:
    return type(e).__name__ in ("FunctionTimeoutError", "TimeoutError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_stages(cfg: Any, items: list[tuple[Path, str, dict[str, Any]]]) -> list[Any]:
    """Run one stage for several simulations in one container; items are (work dir, input name, options).

    Returns, per item, a CompletedProcess or the exception for that item (TimeoutExpired, FileNotFoundError),
    so one simulation's failure doesn't stop the others. Modal being unavailable raises ModalUnavailable.
    """
    s = settings(cfg)
    gpu = str(items[0][2].get("backend", "CPU")).upper() == "CUDA"
    out: list[Any] = [None] * len(items)
    jobs, slots = [], []
    for k, (work, input_name, opts) in enumerate(items):
        files = {}
        try:
            for name in stage_inputs(opts, input_name):
                path = work / name
                if not path.is_file():
                    raise FileNotFoundError(f"stage input {name} missing from {work}")
                files[name] = path.read_bytes()
        except (FileNotFoundError, ValueError) as e:
            out[k] = e
            continue
        jobs.append({"files": files, "input_name": input_name, "job": uuid.uuid4().hex})
        slots.append(k)
    if not jobs:
        return out
    input_names = {j["input_name"] for j in jobs}
    if len(input_names) != 1:
        raise ValueError(f"a batch runs one stage, got inputs {sorted(input_names)}")

    remote = _function(s, gpu)
    try:
        res = remote(jobs, int(cfg.timeout_s))
    except Exception as e:  # noqa: BLE001
        if _is_timeout(e):
            for k in slots:
                out[k] = subprocess.TimeoutExpired([APP_NAME, jobs[0]["input_name"]], cfg.timeout_s)
            return out
        if type(e).__module__.startswith("modal"):
            raise ModalUnavailable(f"Modal call failed ({type(e).__name__}: {e}). Is the app deployed "
                                   f"(`modal deploy src/synbio_bench/origami/modal_app.py`) and are you "
                                   f"logged in (`modal token new`)?") from e
        raise

    vol = _volume(s)
    fn_name = GPU_FUNCTION if gpu else CPU_FUNCTION
    for k, job, r in zip(slots, jobs, res["results"]):
        work = items[k][0]
        try:
            for name in r.get("files", {}):
                with open(work / name, "wb") as f:
                    vol.read_file_into_fileobj(f"{JOBS_DIR}/{job['job']}/{name}", f)
        finally:
            try:
                vol.remove_file(f"{JOBS_DIR}/{job['job']}", recursive=True)
            except Exception:  # noqa: BLE001 - leftovers only cost storage; `modal volume rm` cleans them up
                pass
        if r.get("timed_out"):
            out[k] = subprocess.TimeoutExpired([APP_NAME, job["input_name"]], cfg.timeout_s)
            continue
        header = (f"[modal] job={job['job']} function={fn_name} gpu={res.get('gpu')} oxdna={res.get('oxdna_ref')} "
                  f"remote_seconds={r.get('seconds')} shared_with={len(jobs) - 1} mps={res.get('mps')}\n")
        out[k] = subprocess.CompletedProcess(args=["modal", s["app"], job["input_name"]],
                                             returncode=int(r["returncode"]),
                                             stdout=header + r.get("stdout", ""), stderr=r.get("stderr", ""))
    return out


def run_stage(cfg: Any, work: Path, input_name: str, opts: dict[str, Any]) -> subprocess.CompletedProcess:
    """One simulation's stage (see run_stages)."""
    (r,) = run_stages(cfg, [(work, input_name, opts)])
    if isinstance(r, BaseException):
        raise r
    return r
