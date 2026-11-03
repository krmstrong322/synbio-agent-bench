"""Modal app that runs single oxDNA stages on a cloud GPU (or CPU) for the grader.

Deploy once (and again after changing this file):

    uv sync --extra modal
    uv run modal deploy src/synbio_bench/origami/modal_app.py

The benchmark calls the deployed functions from `modal_runner.py` when `configs/oxdna.yaml` has
`runner.mode: modal` (or `OXDNA_MODE=modal`). Each call runs one stage: the client sends the stage's input
files, the container runs oxDNA in a scratch directory, writes the files the stage produced to a Modal Volume
(trajectories are too big to return inline), and returns the exit code and logs. The client downloads the
outputs into the local work dir and deletes them from the Volume, so analysis stays local and unchanged.

This module must not import anything from `synbio_bench`: Modal loads it on its own inside the container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "synbio-oxdna"
VOLUME_NAME = "synbio-oxdna-runs"
VOLUME_MOUNT = "/vol"
JOBS_DIR = "jobs"
GPU_FUNCTION = "run_stage_gpu"  # must match the function names below
CPU_FUNCTION = "run_stage_cpu"

# Pin oxDNA so every graded run uses the same simulator build. The local WSL build tracks master;
# check `oxDNA` there prints the same version before comparing local and Modal results.
OXDNA_REF = os.environ.get("OXDNA_REF", "v3.7.0")
# Default GPU baked in at deploy time. configs/oxdna.yaml `modal.gpu` overrides it per call.
DEFAULT_GPU = os.environ.get("OXDNA_MODAL_GPU", "T4")
# Upper bound for one stage. The client passes the config's timeout_s, which is enforced inside the container.
MAX_STAGE_SECONDS = 6 * 3600
OXDNA = "/usr/local/bin/oxDNA"

# CUDA_COMMON_ARCH=ON (oxDNA's default) compiles for every architecture nvcc supports, so the image builds
# without a GPU and runs on any Modal GPU type. CUDA 12.8 is the first toolkit with Blackwell (B200).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "git")
    .run_commands(
        f"git clone --depth 1 --branch {OXDNA_REF} https://github.com/lorenzo-rovigatti/oxDNA.git /opt/oxDNA",
        "cmake -S /opt/oxDNA -B /opt/oxDNA/build -DCMAKE_BUILD_TYPE=Release -DCUDA=ON -DCUDA_COMMON_ARCH=ON",
        "cmake --build /opt/oxDNA/build --target oxDNA -j 8",
        f"install -m 755 /opt/oxDNA/build/bin/oxDNA {OXDNA}",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _text(x: str | bytes | None) -> str:
    return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")


def run_stage(files: dict[str, bytes], input_name: str, job: str, timeout_s: int) -> dict:
    """Run `oxDNA <input_name>` on `files`; store produced/changed files under jobs/<job>/ in the Volume."""
    for name in files:
        if "/" in name or name in ("", ".", ".."):
            raise ValueError(f"stage files must be plain names, got {name!r}")
    work = Path(tempfile.mkdtemp(prefix="oxdna-"))
    try:
        for name, data in files.items():
            (work / name).write_bytes(data)
        t0 = time.time()
        timed_out = False
        try:
            proc = subprocess.run([OXDNA, input_name], cwd=work, capture_output=True, text=True,
                                  timeout=min(int(timeout_s), MAX_STAGE_SECONDS))
            returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            returncode, stdout, stderr, timed_out = -9, _text(e.stdout), _text(e.stderr), True
        seconds = round(time.time() - t0, 1)

        produced = {}
        dest = Path(VOLUME_MOUNT) / JOBS_DIR / job
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(work.iterdir()):
            if not p.is_file():
                continue
            if p.name in files and p.read_bytes() == files[p.name]:
                continue  # unchanged input
            shutil.copyfile(p, dest / p.name)
            produced[p.name] = p.stat().st_size
        volume.commit()
        return {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "files": produced,
            "seconds": seconds,
            "gpu": _gpu_name(),
            "oxdna_ref": OXDNA_REF,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.function(gpu=DEFAULT_GPU, volumes={VOLUME_MOUNT: volume}, timeout=MAX_STAGE_SECONDS + 600, retries=1)
def run_stage_gpu(files: dict[str, bytes], input_name: str, job: str, timeout_s: int) -> dict:
    return run_stage(files, input_name, job, timeout_s)


# MC relaxation always uses oxDNA's CPU backend; don't pay GPU rates for it.
@app.function(cpu=2.0, volumes={VOLUME_MOUNT: volume}, timeout=MAX_STAGE_SECONDS + 600, retries=1)
def run_stage_cpu(files: dict[str, bytes], input_name: str, job: str, timeout_s: int) -> dict:
    return run_stage(files, input_name, job, timeout_s)
