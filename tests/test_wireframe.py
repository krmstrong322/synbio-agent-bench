"""Wireframe-shape scoring and the reference wireframe tetrahedron."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from origami_agent.design import load_design, nucleotide_map
from synbio_bench.origami import wireframe
from synbio_bench.origami.spec import validate_design

ROOT = Path(__file__).resolve().parents[1]
TARGET = {"type": "tetrahedron", "edge_nm": 35.0}


@pytest.fixture(scope="module")
def tetrahedron(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("ref") / "tet.sc"
    subprocess.run([sys.executable, "wireframe_tetrahedron.py", str(out)], cwd=ROOT / "reference_designs", check=True)
    return out


@pytest.mark.parametrize("kind,n_vertices,n_edges", [("tetrahedron", 4, 6), ("cube", 8, 12), ("octahedron", 6, 12)])
def test_polyhedra_have_equal_edges(kind, n_vertices, n_edges):
    v, edges = wireframe.polyhedron(kind, 20.0)
    assert (len(v), len(edges)) == (n_vertices, n_edges)
    assert np.allclose([np.linalg.norm(v[i] - v[j]) for i, j in edges], 20.0)


def test_reference_tetrahedron_meets_its_task(tetrahedron):
    spec = yaml.safe_load((ROOT / "tasks" / "origami-wireframe" / "tetrahedron.yaml").read_text())["spec"]
    r = validate_design(tetrahedron, spec)
    assert r["ok"], r["errors"]
    assert r["checks"]["wireframe_footprint"] > 0.98


def test_fit_ignores_rotation_and_translation(tetrahedron):
    pts = wireframe.design_points_nm(load_design(tetrahedron))
    a, b, c = np.radians([30, 50, 70])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    moved = pts @ (rz @ ry @ rx).T + [40.0, -7.0, 3.0]
    assert wireframe.fit_wireframe(moved, TARGET)["f1"] > 0.98


def test_missing_edge_and_rod_score_low(tetrahedron):
    design = load_design(tetrahedron)
    pts = wireframe.design_points_nm(design)
    on_last_edge = [i for (h, *_), i in nucleotide_map(design).index_of.items() if h in (10, 11)]
    fit = wireframe.fit_wireframe(np.delete(pts, on_last_edge, axis=0), TARGET)
    assert min(fit["edge_recall"]) < 0.1 and fit["f1"] < 0.95
    rod = np.c_[np.zeros(1000), np.zeros(1000), np.linspace(-50, 50, 1000)]
    assert wireframe.fit_wireframe(rod, TARGET)["f1"] < 0.5
