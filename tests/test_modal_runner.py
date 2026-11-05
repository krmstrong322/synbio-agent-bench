"""mode: modal without a Modal account: the client is driven against an in-process fake of the deployed app.

The fake mirrors modal_app.run_stages (inputs in, changed files to a volume, metadata back) and runs the mock
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
        self.calls: list[dict] = []  # one per simulation-stage
        self.batches: list[tuple[str, bool, int]] = []  # one per container call: (input, gpu, n jobs)
        self.running = self.peak = 0
        self.lock = threading.Lock()

    def function(self, settings, gpu):
        def remote(jobs, timeout_s):
            with self.lock:
                self.batches.append((jobs[0]["input_name"], gpu, len(jobs)))
                for j in jobs:
                    self.calls.append({"gpu": gpu, "input": j["input_name"], "files": sorted(j["files"])})
                self.running += 1
                self.peak = max(self.peak, self.running)
            try:
                time.sleep(self.delay)
                results = [self._run(j["files"], j["input_name"], j["job"]) for j in jobs]
                return {"results": results, "gpu": "FakeGPU" if gpu else None,
                        "mps": "on" if gpu and len(jobs) > 1 else None, "oxdna_ref": "fake"}
            finally:
                with self.lock:
                    self.running -= 1
        return remote

    def _run(self, files, input_name, job):
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
                    "timed_out": False, "files": produced, "seconds": 0.0}


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
    for name in ("APP_NAME", "VOLUME_NAME", "JOBS_DIR", "GPU_FUNCTION", "CPU_FUNCTION",
                 "BENCH_GPU_FUNCTION", "BENCH_CPU_FUNCTION"):
        assert consts[name] == getattr(modal_runner, name), name
    assert {modal_runner.GPU_FUNCTION, modal_runner.CPU_FUNCTION,
            modal_runner.BENCH_GPU_FUNCTION, modal_runner.BENCH_CPU_FUNCTION} <= funcs


def test_image_is_portable_across_cpus():
    src = (ROOT / "src/synbio_bench/origami/modal_app.py").read_text()
    assert "-DNATIVE_COMPILATION=OFF" in src  # -march=native crashed with SIGILL on Modal (exit -4)
    assert oxdna._signal_note(-4) == " (SIGILL)" and oxdna._signal_note(1) == ""


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


def test_seeds_share_a_gpu_on_modal(tmp_path, monkeypatch):
    """4 seeds, 3 per GPU: a group of 3 and a group of 1, each stage one container call, groups side by side."""
    vol = FakeVolume()
    app = FakeApp(vol, delay=0.2)
    monkeypatch.setattr(modal_runner, "_function", app.function)
    monkeypatch.setattr(modal_runner, "_volume", lambda s: vol)
    g = evaluate_submission(build_reference(tmp_path), HB6_SPEC, modal_cfg(max_parallel=2), tmp_path / "eval",
                            protocol="quick", seeds=[1, 2, 3, 4])
    assert g["valid"] and g["sim_ok"] == [True] * 4
    assert g["sim_runner"] == "modal"
    prod = sorted(n for inp, gpu, n in app.batches if inp == "input_prod")
    assert prod == [1, 3]
    assert ("input_relax_mc", False, 3) in app.batches  # MC relaxation of a group shares one CPU container
    assert app.peak > 1  # the two groups ran at the same time
    log = (tmp_path / "eval/seed_2/log_prod.txt").read_text()
    assert "shared_with=2 mps=on" in log
    assert json.loads((tmp_path / "eval/seed_1/result.json").read_text())["shared_gpu_with"] == 2
    assert "shared_gpu_with" not in json.loads((tmp_path / "eval/seed_4/result.json").read_text())


def test_one_failed_seed_does_not_stop_the_group(tmp_path, monkeypatch):
    vol = FakeVolume()
    app = FakeApp(vol)
    real = app._run

    def flaky(files, input_name, job):
        out = real(files, input_name, job)
        if input_name == "input_prod" and b"seed = 2" in files[input_name]:
            return {**out, "returncode": 1, "stdout": "boom", "files": {}}
        return out

    app._run = flaky
    monkeypatch.setattr(modal_runner, "_function", app.function)
    monkeypatch.setattr(modal_runner, "_volume", lambda s: vol)
    design = build_reference(tmp_path)
    rs = oxdna.simulate_many([oxdna.SimJob(design, tmp_path / f"s{k}", k) for k in (1, 2, 3)], modal_cfg(),
                             protocol="quick")
    assert [r.get("ok") for r in rs] == [True, False, True]
    assert rs[1]["failed_stage"] == "prod" and "boom" in rs[1]["log_tail"]


def test_local_runs_stay_one_at_a_time(tmp_path):
    cfg = OxDNAConfig(mode="mock")
    assert cfg.sims_per_gpu == 1 and cfg.max_parallel == 1
    assert modal_cfg().sims_per_gpu == 3 and modal_cfg(sims_per_gpu=1).sims_per_gpu == 1
    design = build_reference(tmp_path)
    rs = oxdna.simulate_many([oxdna.SimJob(design, tmp_path / f"s{k}", k) for k in (1, 2)], cfg, protocol="quick")
    assert all(r["ok"] for r in rs) and [r["seed"] for r in rs] == [1, 2]
    assert all("shared_gpu_with" not in r for r in rs)


def test_container_side_run_stages(tmp_path, monkeypatch):
    """modal_app.run_stages against a stand-in oxDNA: per-job logs and changed files; MPS failure -> one at a time."""
    pytest.importorskip("modal")
    from synbio_bench.origami import modal_app

    fake = tmp_path / "oxDNA"
    fake.write_text(f"#!{sys.executable}\n"
                    "import sys, pathlib, time\n"
                    "print('running', sys.argv[1], pathlib.Path('tag').read_text())\n"
                    "time.sleep(1.5)\n"
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
    monkeypatch.setattr(modal_app, "_gpu_name", lambda: "FakeGPU")
    monkeypatch.setattr(modal_app, "_start_mps", lambda root: "nvidia-cuda-mps-control not found")

    def jobs(n):
        return [{"files": {"input_x": b"steps = 1\n", "initial.top": b"top", "relaxed.dat": b"old",
                           "tag": f"seed{k}".encode()}, "input_name": "input_x", "job": f"job{k}"} for k in range(n)]

    t0 = time.time()
    out = modal_app.run_stages(jobs(2), timeout_s=60, gpu=False)  # CPU: side by side
    assert time.time() - t0 < 2.7  # one after another would take > 3 s
    assert out["mps"] is None and out["gpu"] is None
    r0, r1 = out["results"]
    assert r0["returncode"] == 0 and "running input_x seed0" in r0["stdout"] and "seed1" in r1["stdout"]
    assert set(r0["files"]) == {"relaxed.dat", "last_conf.dat"}  # unchanged inputs and the log stay behind
    assert (tmp_path / "vol/jobs/job1/relaxed.dat").read_text() == "moved"

    t0 = time.time()
    out = modal_app.run_stages(jobs(2), timeout_s=60, gpu=True)  # GPU, MPS unavailable: one at a time
    assert time.time() - t0 >= 3.0
    assert out["mps"].startswith("off (nvidia-cuda-mps-control not found)") and out["gpu"] == "FakeGPU"
    assert Vol.commits == 2
    with pytest.raises(ValueError):
        modal_app.run_stages([{"files": {"../x": b""}, "input_name": "../x", "job": "j"}], timeout_s=60, gpu=False)


def test_default_gpus_fall_back():
    src = (ROOT / "src/synbio_bench/origami/modal_app.py").read_text()
    assert '"OXDNA_MODAL_GPU", "L4,A10G"' in src


def test_env_override_sets_modal_gpu(monkeypatch):
    monkeypatch.setenv("OXDNA_MODAL_GPU", "L4")
    cfg = oxdna.config_from_env(OxDNAConfig(mode="modal", modal={"max_parallel": 2}))
    assert cfg.modal == {"max_parallel": 2, "gpu": "L4"}


# ---------------------------------------------------------------------------
# GPU benchmark (scripts/modal_gpu_benchmark.py and modal_app.bench)
# ---------------------------------------------------------------------------


def load_bench_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("modal_gpu_benchmark", ROOT / "scripts/modal_gpu_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_variants_and_costs():
    b = load_bench_script()
    cfg = OxDNAConfig(mode="modal")
    vs = b.variants_for(cfg, "T4", 1000, [0.5, 0.2], [1, 3], mps=True)
    assert [v["name"] for v in vs] == ["skin0.5-x1", "skin0.5-x3", "skin0.5-x3-mps",
                                       "skin0.2-x1", "skin0.2-x3", "skin0.2-x3-mps"]
    opts = parse_input(vs[1]["inputs"][2])
    assert opts["verlet_skin"] == "0.5" and opts["seed"] == "3" and opts["conf_file"] == "relaxed.dat"
    assert opts["backend"] == "CUDA" and opts["steps"] == "1000"
    cpu = parse_input(b.variants_for(cfg, "CPU", 100, [0.2], [1], mps=True)[0]["inputs"][0])
    assert cpu["backend"] == "CPU" and "max_density_multiplier" not in cpu

    result = {"gpu": "Tesla T4", "container_s": 100.0, "variants": [
        {"name": "skin0.5-x1", "copies": 1, "mps": False, "wall_s": 20.0,
         "runs": [{"returncode": 0, "timed_out": False, "per_step_ms": 0.5, "last_energy": "1000 -1.40 0.1 -1.3"}]},
        {"name": "skin0.5-x3", "copies": 3, "mps": False, "wall_s": 30.0,
         "runs": [{"returncode": 0, "timed_out": False, "per_step_ms": 0.75, "last_energy": "1000 -1.42 0.1 -1.3"}] * 3},
        {"name": "skin0.5-x3-mps", "copies": 3, "mps": True, "skipped": "MPS unavailable: missing"},
    ]}
    rows = b.rows_from("triangle", 4000, "T4", 1000, result, vs)
    one, three, mps = rows
    price = b.price_per_s("T4")
    assert one["usd_per_seed_shape"] == pytest.approx(price * 0.5e-3 * 1.2e6, rel=1e-3)
    assert three["usd_per_seed_shape"] == pytest.approx(price * 0.75e-3 * 1.2e6 / 3, rel=1e-3)  # sharing pays
    assert one["e_pot_mean"] == -1.4 and three["skin"] == 0.5
    assert mps["ok"] is False and "MPS" in mps["note"]
    md = b.summary_md(rows, spent=0.12)
    assert "Cheapest: **T4**, skin 0.5, 3 per GPU" in md and "$0.12" in md
    with pytest.raises(ValueError):
        b.price_per_s("V100")


def test_container_side_bench(tmp_path, monkeypatch):
    """modal_app.bench runs copies side by side, parses oxDNA's per-step time, and skips MPS it can't start."""
    pytest.importorskip("modal")
    from synbio_bench.origami import modal_app

    fake = tmp_path / "oxDNA"
    fake.write_text(f"#!{sys.executable}\n"
                    "import pathlib, sys, time\n"
                    "assert pathlib.Path('relaxed.dat').read_text() == 'conf'\n"
                    "time.sleep(1.0)\n"
                    "pathlib.Path('energy.dat').write_text('0 -1.3 0.1 -1.2\\n100 -1.4 0.1 -1.3\\n')\n"
                    "print('INFO: Total Running Time: 0.3 s, per step: 0.0421 ms', file=sys.stderr)\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(modal_app, "OXDNA", str(fake))
    monkeypatch.setattr(modal_app, "_start_mps", lambda root: "nvidia-cuda-mps-control not found")
    out = modal_app.bench({"relaxed.dat": b"conf"}, [
        {"name": "x2", "inputs": ["steps = 100\n"] * 2, "mps": False},
        {"name": "x2-mps", "inputs": ["steps = 100\n"] * 2, "mps": True},
    ], timeout_s=60)
    x2, x2mps = out["variants"]
    assert [r["per_step_ms"] for r in x2["runs"]] == [0.0421, 0.0421]
    assert all(r["returncode"] == 0 and r["last_energy"].startswith("100 ") for r in x2["runs"])
    assert x2["wall_s"] < 1.8  # the copies overlapped (one after the other would take > 2 s)
    assert x2mps["skipped"].startswith("MPS unavailable")


def test_bench_script_end_to_end(tmp_path, monkeypatch, fake_modal):
    """main(): relax on (fake) Modal, save after each container, survive a GPU that never comes, resume."""
    b = load_bench_script()
    seen: dict = {"calls": []}

    def fake_call(gpu, files, variants, timeout_s, environment, wait_s):
        seen["files"] = sorted(files)
        seen["calls"].append(gpu)
        if gpu == "L4" and not seen.get("l4_free"):
            raise b.NoResult("no result after 30 min (no L4 free?); call cancelled")
        return {"gpu": f"Fake {gpu}", "container_s": 12.0, "variants": [
            {"name": v["name"], "copies": len(v["inputs"]), "mps": v["mps"], "wall_s": 5.0,
             "runs": [{"returncode": 0, "timed_out": False, "per_step_ms": 0.2,
                       "last_energy": "30000 -1.41 0.1 -1.3"}] * len(v["inputs"])} for v in variants]}

    monkeypatch.setattr(b, "call_bench", fake_call)
    out = tmp_path / "bench"
    argv = ["bench", "--design", "tetrahedron", "--gpu", "T4", "--gpu", "L4", "--skin", "0.2",
            "--copies", "1", "--copies", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    b.main()
    assert seen["files"] == ["designed_pairs.op", "initial.top", "relaxed.dat"]
    rows = json.loads((out / "results.json").read_text())["rows"]
    t4 = [r for r in rows if r["gpu"] == "T4"]
    assert [(r["copies"], r["ok"], r["skin"]) for r in t4] == [(1, True, 0.2), (2, True, 0.2)]
    assert t4[0]["n_nt"] == 2088 == b.DESIGNS["tetrahedron"][2]
    l4 = [r for r in rows if r["gpu"] == "L4"]
    assert len(l4) == 1 and not l4[0]["ok"] and "no L4 free" in l4[0]["note"]
    summary = (out / "summary.md").read_text()
    assert "Cheapest: **T4**, skin 0.2, 2 per GPU" in summary and "no L4 free" in summary
    assert (out / "results.csv").read_text().startswith("design,")

    # Rerun: T4 is kept, only the failed L4 pair is retried, relaxation is cached.
    relax_calls = len(fake_modal.calls)
    seen.update(calls=[], l4_free=True)
    b.main()
    assert seen["calls"] == ["L4"]
    assert len(fake_modal.calls) == relax_calls
    rows = json.loads((out / "results.json").read_text())["rows"]
    assert {(r["gpu"], r["copies"]) for r in rows if r["ok"]} == {("T4", 1), ("T4", 2), ("L4", 1), ("L4", 2)}
