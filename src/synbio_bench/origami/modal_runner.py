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
import uuid
from pathlib import Path
from typing import Any, Callable

# Must match modal_app.py (tests/test_modal_runner.py checks this without importing modal).
APP_NAME = "synbio-oxdna"
VOLUME_NAME = "synbio-oxdna-runs"
JOBS_DIR = "jobs"
GPU_FUNCTION = "run_stage_gpu"
CPU_FUNCTION = "run_stage_cpu"

DEFAULTS: dict[str, Any] = {
    "app": APP_NAME,
    "volume": VOLUME_NAME,
    "gpu": None,  # None: the GPU the app was deployed with (T4 unless OXDNA_MODAL_GPU was set at deploy)
    "max_parallel": 4,  # concurrent simulations; Modal's Starter plan allows ~10 GPUs at once
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


def _function(s: dict[str, Any], gpu: bool) -> Callable[..., dict]:
    modal = _modal()
    fn = modal.Function.from_name(s["app"], GPU_FUNCTION if gpu else CPU_FUNCTION,
                                  environment_name=s["environment"])
    if gpu and s["gpu"]:
        fn = fn.with_options(gpu=s["gpu"])
    return fn.remote


def _volume(s: dict[str, Any]) -> Any:
    return _modal().Volume.from_name(s["volume"], environment_name=s["environment"])


def _is_timeout(e: Exception) -> bool:
    return type(e).__name__ in ("FunctionTimeoutError", "TimeoutError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_stage(cfg: Any, work: Path, input_name: str, opts: dict[str, Any]) -> subprocess.CompletedProcess:
    s = settings(cfg)
    gpu = str(opts.get("backend", "CPU")).upper() == "CUDA"
    files = {}
    for name in stage_inputs(opts, input_name):
        path = work / name
        if not path.is_file():
            raise FileNotFoundError(f"stage input {name} missing from {work}")
        files[name] = path.read_bytes()

    job = uuid.uuid4().hex
    remote = _function(s, gpu)
    try:
        out = remote(files, input_name, job, int(cfg.timeout_s))
    except Exception as e:  # noqa: BLE001
        if _is_timeout(e):
            raise subprocess.TimeoutExpired([APP_NAME, input_name], cfg.timeout_s) from e
        if type(e).__module__.startswith("modal"):
            raise ModalUnavailable(f"Modal call failed ({type(e).__name__}: {e}). Is the app deployed "
                                   f"(`modal deploy src/synbio_bench/origami/modal_app.py`) and are you "
                                   f"logged in (`modal token new`)?") from e
        raise

    vol = _volume(s)
    try:
        for name in out.get("files", {}):
            with open(work / name, "wb") as f:
                vol.read_file_into_fileobj(f"{JOBS_DIR}/{job}/{name}", f)
    finally:
        try:
            vol.remove_file(f"{JOBS_DIR}/{job}", recursive=True)
        except Exception:  # noqa: BLE001 - leftovers only cost storage; `modal volume rm` cleans them up
            pass

    if out.get("timed_out"):
        raise subprocess.TimeoutExpired([APP_NAME, input_name], cfg.timeout_s)
    header = (f"[modal] job={job} function={GPU_FUNCTION if gpu else CPU_FUNCTION} gpu={out.get('gpu')} "
              f"oxdna={out.get('oxdna_ref')} remote_seconds={out.get('seconds')}\n")
    return subprocess.CompletedProcess(args=["modal", s["app"], input_name], returncode=int(out["returncode"]),
                                       stdout=header + out.get("stdout", ""), stderr=out.get("stderr", ""))
