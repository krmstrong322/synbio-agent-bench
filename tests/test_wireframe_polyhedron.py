"""The general wireframe reference builder (reference_designs/wireframe_polyhedron.py): cube and octahedron.

No oxDNA: checks the design-time geometry only (a single scaffold tracing the target, passing the task spec).
The simulated shape is a separate, GPU-bound check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference_designs"))
_spec = importlib.util.spec_from_file_location("wireframe_polyhedron", ROOT / "reference_designs/wireframe_polyhedron.py")
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)

from origami_agent.design import summarize_design  # noqa: E402
from synbio_bench.origami.spec import validate_design  # noqa: E402
from synbio_bench.origami.wireframe import design_points_nm, fit_wireframe, polyhedron  # noqa: E402


@pytest.mark.parametrize("kind,n_vertices,n_edges", [("cube", 8, 12), ("octahedron", 6, 12)])
def test_reference_traces_the_target(tmp_path, kind, n_vertices, n_edges):
    design = wp.create_design(kind)
    # edges match the grader's polyhedron() exactly, so the fit target lines up
    _, grader_edges = polyhedron(kind, 35.0)
    assert len(wp.Polyhedron(kind).edges) == len(grader_edges) == n_edges

    sm = summarize_design(design)
    assert sm["num_helices"] == 2 * n_edges
    assert sm["helix_connectivity_components"] == 1
    assert sm["mismatched_pairs"] == 0 and sm["unassigned_bases"] == 0
    scaffolds = [s for s in design.strands if s.is_scaffold]
    assert len(scaffolds) == 1  # one scaffold routed through the whole frame
    # every edge's both helices are visited by the scaffold
    scaf_helices = {d.helix for d in scaffolds[0].domains if hasattr(d, "helix")}
    assert len(scaf_helices) == 2 * n_edges

    fit = fit_wireframe(design_points_nm(design), {"type": kind, "edge_nm": 35.0, "tube_nm": 3.5})
    assert fit["f1"] >= 0.95, fit


@pytest.mark.parametrize("kind", ["cube", "octahedron"])
def test_reference_passes_its_task_spec(kind):
    spec = yaml.safe_load((ROOT / f"tasks/origami-wireframe/{kind}.yaml").read_text())["spec"]
    design = wp.create_design(kind)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"{kind}.sc"
        design.write_scadnano_file(filename=path.name, directory=d)
        v = validate_design(path, spec)
    assert v["ok"], v["errors"]
    assert all(score == 1.0 for score in v["checks"].values()), v["checks"]


def test_spanning_tree_covers_all_vertices():
    poly = wp.Polyhedron("octahedron")
    assert len(poly.tree) == len(poly.V) - 1  # a tree on n vertices has n-1 edges
    in_tree = {v for k in poly.tree for v in poly.edges[k]}
    assert in_tree == set(range(len(poly.V)))
