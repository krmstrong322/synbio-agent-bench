"""Shape targets and shape-fidelity scoring for flat origami tasks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from synbio_bench.origami import shapes
from synbio_bench.origami.design import load_design, validate_design
from synbio_bench.origami.oxdna import OxDNAConfig, simulate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["rectangle", "triangle", "disk", "star", "hexagon"])
def test_polygons_have_unit_area_and_match_themselves(kind):
    poly = shapes.shape_polygon({"type": kind})
    assert shapes.polygon_area(poly) == pytest.approx(1.0)
    g = np.stack(np.meshgrid(np.arange(-50, 50) + 0.5, np.arange(-50, 50) + 0.5), -1).reshape(-1, 2)
    cells = g[shapes.points_in_polygon(g, poly * 50)]
    assert shapes.best_iou(cells, {"type": kind})["iou"] > 0.95  # pixel-edge error at 50 px scale


def test_wrong_shape_scores_low():
    poly = shapes.shape_polygon({"type": "triangle"}) * 50
    g = np.stack(np.meshgrid(np.arange(-50, 50) + 0.5, np.arange(-50, 50) + 0.5), -1).reshape(-1, 2)
    cells = g[shapes.points_in_polygon(g, poly)]
    assert shapes.best_iou(cells, {"type": "disk"})["iou"] < 0.75


@pytest.fixture(scope="module")
def rothemund_rect(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ref") / "rect.sc"
    subprocess.run([sys.executable, "rothemund.py", "rect", str(out)], cwd=ROOT / "reference_designs", check=True)
    return out


def test_rothemund_reference_meets_its_task(rothemund_rect):
    spec = yaml.safe_load((ROOT / "tasks/origami-rothemund/rectangle.yaml").read_text())["spec"]
    r = validate_design(rothemund_rect, spec)
    assert r["ok"], r["errors"]
    assert r["checks"]["shape_footprint"] > 0.98


def test_rectangle_reference_fails_triangle_task(rothemund_rect):
    spec = yaml.safe_load((ROOT / "tasks/origami-shapes/triangle-7000.yaml").read_text())["spec"]
    r = validate_design(rothemund_rect, spec)
    assert any("IoU" in e for e in r["errors"])
    assert r["checks"]["shape_footprint"] < 0.75


def test_non_planar_design_has_no_footprint(tmp_path):
    hb6 = tmp_path / "hb6.sc"
    subprocess.run([sys.executable, "hb6_honeycomb.py", str(hb6)], cwd=ROOT / "reference_designs", check=True)
    assert shapes.footprint_cells(load_design(hb6)) is None


def test_mock_simulation_reports_shape_fidelity(rothemund_rect, tmp_path):
    shape = {"type": "rectangle", "aspect": 1.5692}
    r = simulate(rothemund_rect, tmp_path / "sim", OxDNAConfig(mode="mock"), protocol="quick", shape=shape)
    assert r["ok"], r
    fid = r["metrics"]["shape_fidelity"]
    assert fid["iou"] > 0.85 and fid["flatness"] < 0.2


def test_figures_render_for_a_mock_run(rothemund_rect, tmp_path):
    from synbio_bench.origami import viz

    run = tmp_path / "run"
    (run / "workspace" / "final").mkdir(parents=True)
    (run / "workspace" / "final" / "design.sc").write_bytes(rothemund_rect.read_bytes())
    shape = {"type": "rectangle", "aspect": 1.5692}
    r = simulate(rothemund_rect, run / "eval" / "seed_1", OxDNAConfig(mode="mock"), protocol="quick", shape=shape)
    assert r["ok"]
    figs = viz.figures_for_run(run, {"shape": shape}, "test")
    assert set(figs) == {"design", "structure_occupancy", "structure_strands", "sim_dir"}
    svg = (run / figs["design"]).read_text(encoding="utf-8")
    assert svg.startswith("<svg") and svg.count("<path") > 200
    assert (run / figs["structure_occupancy"]).stat().st_size > 10_000
