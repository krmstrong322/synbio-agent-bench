"""Shape targets and shape-fidelity scoring for flat origami tasks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from synbio_bench.origami import shapes
from origami_agent.design import load_design
from synbio_bench.origami.spec import validate_design
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


def _sheet_points(design_path, tmp_path):
    """Exported (ideal, flat) nucleotide positions in nm, plus slots and backbone links."""
    import json

    from synbio_bench.origami import analysis
    from synbio_bench.origami.oxdna import export_design

    export_design(design_path, tmp_path / "ox")
    _, body = analysis.read_conf(tmp_path / "ox" / "initial.dat")
    _, n3 = analysis.read_topology(tmp_path / "ox" / "initial.top")
    slots = json.loads((tmp_path / "ox" / "index_map.json").read_text())["slots"]
    return shapes.project_to_plane(body[:, :3] * 0.8518), slots, n3


def test_unrolled_score_ignores_rolling_but_catches_tearing(rothemund_rect, tmp_path):
    shape = {"type": "rectangle", "aspect": 1.5692}
    flat, slots, n3 = _sheet_points(rothemund_rect, tmp_path)
    # roll the sheet into a tube: the across-helix direction wraps around a cylinder
    width = np.ptp(flat[:, 1])
    r = width / (2 * np.pi * 0.95)
    theta = (flat[:, 1] - flat[:, 1].min()) / r
    rolled = np.c_[flat[:, 0], r * np.sin(theta), r * (1 - np.cos(theta))]

    flat_fit, _ = shapes.unrolled_shape_iou(flat, slots, n3, shape)
    rolled_fit, _ = shapes.unrolled_shape_iou(rolled, slots, n3, shape)
    assert abs(rolled_fit["iou"] - flat_fit["iou"]) < 0.03
    assert shapes.simulated_shape_iou(rolled, shape)["iou"] < 0.7  # the projected outline is fooled

    # tear the sheet: pull the upper half of the helices 20 nm away from the rest
    helix = np.array([s[0] if s else -1 for s in slots])
    torn = flat.copy()
    torn[helix >= 12, 2] += 20
    torn_fit, _ = shapes.unrolled_shape_iou(torn, slots, n3, shape)
    assert torn_fit["components"] >= 2 and torn_fit["iou"] < flat_fit["iou"] - 0.2


@pytest.fixture(scope="module")
def sharp_triangle(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ref") / "tri.sc"
    subprocess.run([sys.executable, "sharp_triangle.py", str(out)], cwd=ROOT / "reference_designs", check=True)
    return out


def test_sharp_triangle_reference_meets_its_task(sharp_triangle):
    """Three trapezoids in rotated helix groups count as one flat sheet, and the open centre is filled."""
    spec = yaml.safe_load((ROOT / "tasks/origami-shapes/triangle-4000.yaml").read_text(encoding="utf-8"))["spec"]
    d = load_design(sharp_triangle)
    assert len({h.group for h in d.helices.values()}) == 3 and shapes.planar_sheet(d)
    r = validate_design(sharp_triangle, spec)
    assert r["ok"], r["errors"]
    assert r["checks"]["shape_footprint"] > 0.95


def test_tilted_helix_group_is_not_planar(sharp_triangle):
    d = load_design(sharp_triangle)
    d.groups["B"].yaw = 30  # rotate one trapezoid out of the sheet's plane
    assert not shapes.planar_sheet(d)
    assert shapes.footprint_cells(d) is None


def test_fill_holes_closes_an_annulus():
    g = np.stack(np.meshgrid(np.arange(-20, 20) + 0.5, np.arange(-20, 20) + 0.5), -1).reshape(-1, 2)
    r = np.linalg.norm(g, axis=1)
    ring = g[(r < 18) & (r > 9)]
    assert len(shapes.fill_holes(ring)) == np.count_nonzero(r < 18)


def test_unrolled_hollow_triangle_matches_its_flat_outline(sharp_triangle, tmp_path):
    """Seams are stitched through bridging loops and designed contacts; the hole doesn't distort the layout."""
    shape = {"type": "triangle"}
    flat, slots, n3 = _sheet_points(sharp_triangle, tmp_path)
    loops = [link for link in shapes.crossover_links(slots, n3) if link[2] > 0]
    assert len(loops) >= 10  # bridging staples and scaffold jumps cross the seams through unpaired loops
    fit, _ = shapes.unrolled_shape_iou(flat, slots, n3, shape, flat)
    axis = shapes.helix_axis_points(flat, slots)
    exact = shapes.best_iou(shapes._dilated_cells(np.array(list(axis.values()))[:, :2]), shape)["iou"]
    assert fit["components"] == 1 and abs(fit["iou"] - exact) < 0.02
