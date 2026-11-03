"""mode: modal without a Modal account: the client is driven against an in-process fake of the deployed app.

The fake mirrors modal_app.run_stage (inputs in, changed files to a volume, metadata back) and runs the mock
simulator, so these tests cover file shipping, CPU/GPU routing, timeouts, cleanup and parallel seeds.
The container-side code itself is tested against a stand-in oxDNA when the modal extra is installed.
"""

from __future__ import annotations

import ast
import json
import re
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from synbio_bench.origami import modal_runner, oxdna
from synbio_bench.origami.oxdna import OxDNAConfig, build_inputs, simulate
from synbio_bench.origami.scoring import evaluate_submission

from test_smoke import HB6_SPEC, build_reference

ROOT = Path(__file__).resolve().parents[1]


def parse_input(text: str) -> dict:
    """Top-level `key = value` options of an oxDNA input file (enough for the mock simulator)."""
    opts, depth = {}, 0
    for line in text.splitlines():
        if depth == 0 and " = " in line:
            k, v = line.split(" = ", 1)
            opts[k.strip()] = v.strip()
        depth += line.count("{") - line.count("}")
    return opts


class FakeVolume:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def read_file_into_fileobj(self, path, fileobj):
        fileobj.write(self.files[path])
        return len(self.files[path])

    def remove_file(self, path, recursive=False):
        with self.lock:
            for k in [k for k in self.files if k == path or k.startswith(path + "/")]:
                del self.files[k]


class FakeApp:
    """Stands in for the deployed functions: runs the mock oxDNA stage in a scratch dir."""

    def __init__(self, volume: FakeVolume, delay: float = 0.0, timed_out: bool = False):
        self.volume, self.delay, self.timed_out = volume, delay, timed_out
        self.calls: list[dict] = []
        self.running = self.peak = 0
        self.lock = threading.Lock()

    def function(self, settings, gpu):
        def remote(files, input_name, job, timeout_s):
            with self.lock:
                self.calls.append({"gpu": gpu, "input": input_name, "files": sorted(files)})
                self.running += 1
                self.peak = max(self.peak, self.running)
            try:
                time.sleep(self.delay)
                return self._run(files, input_name, job, gpu)
            finally:
                with self.lock:
                    self.running -= 1
        return remote

    def _run(self, files, input_name, job, gpu):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            for name, data in files.items():
                (work / name).write_bytes(data)
            opts = parse_input(files[input_name].decode())
            if "data_output_1" in opts:  # the mock reads designed pairs from index_map.json
                op = files["designed_pairs.op"].decode()
                pairs = [[int(i), int(j)] for i, j in re.findall(r"pair\d+ = (\d+), (\d+)", op)]
                (work / "index_map.json").write_text(json.dumps({"pairs": pairs}))
            if self.timed_out:
                return {"returncode": -9, "stdout": "", "stderr": "", "timed_out": True, "files": {}}
            proc = oxdna._mock_stage(work, opts)
            produced = {}
            for p in work.iterdir():
                if p.name == "index_map.json" or files.get(p.name) == p.read_bytes():
                    continue
                with self.volume.lock:
                    self.volume.files[f"jobs/{job}/{p.name}"] = p.read_bytes()
                produced[p.name] = p.stat().st_size
            return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                    "timed_out": False, "files": produced, "seconds": 0.0, "gpu": "FakeGPU" if gpu else None,
                    "oxdna_ref": "fake"}


@pytest.fixture
def fake_modal(monkeypatch):
    vol = FakeVolume()
    app = FakeApp(vol)
    monkeypatch.setattr(modal_runner, "_function", app.function)
    monkeypatch.setattr(modal_runner, "_volume", lambda s: vol)
    return app


def modal_cfg(**modal):
    return OxDNAConfig(mode="modal", modal=modal)


def test_names_match_deployed_app():
    tree = ast.parse((ROOT / "src/synbio_bench/origami/modal_app.py").read_text())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("APP_NAME", "VOLUME_NAME", "JOBS_DIR", "GPU_FUNCTION", "CPU_FUNCTION"):
        assert consts[name] == getattr(modal_runner, name), name
    assert {modal_runner.GPU_FUNCTION, modal_runner.CPU_FUNCTION} <= funcs


def test_stage_inputs_are_minimal():
    stages = build_inputs(OxDNAConfig(), "quick")
    assert modal_runner.stage_inputs(stages["relax_mc"], "input_relax_mc") == [
        "input_relax_mc", "initial.top", "initial.dat"]
    assert modal_runner.stage_inputs(stages["prod"], "input_prod") == [
        "input_prod", "initial.top", "relaxed.dat", "designed_pairs.op"]
    with pytest.raises(ValueError):
        modal_runner.stage_inputs({"topology": "/abs/initial.top"}, "input")


def test_max_parallel():
    assert OxDNAConfig(mode="mock").max_parallel == 1
    assert OxDNAConfig(mode="wsl").max_parallel == 1
    assert modal_cfg().max_parallel == modal_runner.DEFAULTS["max_parallel"]
    assert modal_cfg(max_parallel=2).max_parallel == 2


def test_simulate_on_fake_modal(tmp_path, fake_modal):
    r = simulate(build_reference(tmp_path), tmp_path / "sim", modal_cfg(), protocol="quick")
    assert r["ok"], r
    assert r["runner"] == "modal"
    assert 0.9 < r["metrics"]["bp_fraction_mean"] <= 1.0
    # MC relaxation goes to the CPU function, MD stages to the GPU function.
    # Extra relax rounds (relax_md_k, which overwrite relaxed.dat) also run remotely: mock jitter leaves bonds
    # past the FENE margin, so this exercises the local backbone check between remote stages.
    calls = [(c["input"], c["gpu"]) for c in fake_modal.calls]
    assert calls[0] == ("input_relax_mc", False)
    assert calls[1] == ("input_relax_md", True)
    assert calls[-1] == ("input_prod", True)
    assert all(gpu for _, gpu in calls[1:])
    assert "initial.dat" not in fake_modal.calls[-1]["files"]  # only what the stage reads is uploaded
    sim = tmp_path / "sim"
    for f in ("relaxed_mc.dat", "relaxed.dat", "last_conf.dat", "trajectory.dat", "hb_list.txt"):
        assert (sim / f).is_file(), f
    assert (sim / "log_prod.txt").read_text().startswith("[modal] job=")
    assert modal_runner._volume(None).files == {}  # outputs are deleted from the volume after download


def test_remote_timeout_fails_the_stage(tmp_path, monkeypatch):
    vol = FakeVolume()
    app = FakeApp(vol, timed_out=True)
    monkeypatch.setattr(modal_runner, "_function", app.function)
    monkeypatch.setattr(modal_runner, "_volume", lambda s: vol)
    r = simulate(build_reference(tmp_path), tmp_path / "sim", modal_cfg(), protocol="quick")
    assert not r["ok"]
    assert r["failed_stage"] == "relax_mc" and "timed out" in r["error"]


def test_missing_modal_is_not_a_simulation_failure(tmp_path, monkeypatch):
    def unavailable(settings, gpu):
        raise modal_runner.ModalUnavailable("not installed")
    monkeypatch.setattr(modal_runner, "_function", unavailable)
    with pytest.raises(modal_runner.ModalUnavailable):
        simulate(build_reference(tmp_path), tmp_path / "sim", modal_cfg(), protocol="quick")


def test_seeds_run_in_parallel_on_modal(tmp_path, monkeypatch):
    vol = FakeVolume()
    app = FakeApp(vol, delay=0.2)
    monkeypatch.setattr(modal_runner, "_function", app.function)
    monkeypatch.setattr(modal_runner, "_volume", lambda s: vol)
    g = evaluate_submission(build_reference(tmp_path), HB6_SPEC, modal_cfg(max_parallel=3), tmp_path / "eval",
                            protocol="quick", seeds=[1, 2, 3])
    assert g["valid"] and g["sim_ok"] == [True, True, True]
    assert g["sim_runner"] == "modal"
    assert app.peak > 1


def test_container_side_run_stage(tmp_path, monkeypatch):
    """modal_app.run_stage against a stand-in oxDNA: returns logs and only new or changed files."""
    pytest.importorskip("modal")
    from synbio_bench.origami import modal_app

    fake = tmp_path / "oxDNA"
    fake.write_text(f"#!{sys.executable}\n"
                    "import sys, pathlib\n"
                    "print('running', sys.argv[1])\n"
                    "pathlib.Path('relaxed.dat').write_text('moved')\n"
                    "pathlib.Path('last_conf.dat').write_text('new')\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    class Vol:
        commits = 0

        def commit(self):
            Vol.commits += 1

    monkeypatch.setattr(modal_app, "OXDNA", str(fake))
    monkeypatch.setattr(modal_app, "VOLUME_MOUNT", str(tmp_path / "vol"))
    monkeypatch.setattr(modal_app, "volume", Vol())
    files = {"input_x": b"steps = 1\n", "initial.top": b"top", "relaxed.dat": b"old"}
    out = modal_app.run_stage(files, "input_x", "job1", timeout_s=60)
    assert out["returncode"] == 0 and "running input_x" in out["stdout"]
    assert set(out["files"]) == {"relaxed.dat", "last_conf.dat"}
    assert (tmp_path / "vol/jobs/job1/relaxed.dat").read_text() == "moved"
    assert Vol.commits == 1
    with pytest.raises(ValueError):
        modal_app.run_stage({"../x": b""}, "../x", "job2", timeout_s=60)


def test_env_override_sets_modal_gpu(monkeypatch):
    monkeypatch.setenv("OXDNA_MODAL_GPU", "L4")
    cfg = oxdna.config_from_env(OxDNAConfig(mode="modal", modal={"max_parallel": 2}))
    assert cfg.modal == {"max_parallel": 2, "gpu": "L4"}
