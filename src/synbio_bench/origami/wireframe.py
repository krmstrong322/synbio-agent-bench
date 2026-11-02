"""Wireframe-shape fidelity: how well a 3D point cloud traces a target polyhedron's edges.

The target is an ideal polyhedron (vertices, edges) with a given vertex-to-vertex edge length. The structure
(the simulated mean, or the design's own coordinates) is rigidly fitted onto the edge set by ICP from many
starting rotations; then, with a tube of radius `tube_nm` around every target edge:

  precision = fraction of the structure's nucleotides inside a tube (no material where there is no edge)
  recall    = fraction of points sampled along the target edges, away from the vertices, with a nucleotide
              within tube_nm (no missing edge; near a vertex the other edges would cover for a missing one)
  f1        = 2PR / (P + R), the graded score; like the flat shapes' IoU, it is 1 for a perfect trace

The tube (3.5 nm) holds a two-helix edge (helix axes 1.3 nm off the edge line, nucleotides ~1 nm further out)
with ~1 nm to spare, so straight, well-placed edges score near 1 while bowed, missing, collapsed or wrongly
sized edges lose either precision or recall. Nothing depends on how the design is built (edge thickness, routing,
helix groups), only on where its nucleotides end up.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

TUBE_NM = 3.5
SAMPLE_NM = 0.5  # spacing of recall sample points along the target edges
END_NM = 6.0  # recall ignores this much of each edge at either vertex
ICP_STARTS = 64
ICP_ITERS = 40


def polyhedron(kind: str, edge_nm: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Vertices (centred on the origin, nm) and edges of a regular polyhedron with the given edge length."""
    if kind == "tetrahedron":
        v = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float)
    elif kind == "cube":
        v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    elif kind == "octahedron":
        v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], float)
    else:
        raise ValueError(f"unknown wireframe type {kind!r}")
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    shortest = d[d > 1e-9].min()
    edges = [(i, j) for i in range(len(v)) for j in range(i + 1, len(v)) if abs(d[i, j] - shortest) < 1e-6]
    return v * edge_nm / shortest, edges


def _closest_on_edges(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to the nearest segment, and that nearest point."""
    d = b - a  # (m, 3)
    t = np.clip(np.einsum("nmk,mk->nm", p[:, None] - a[None], d) / (d * d).sum(1), 0, 1)
    q = a[None] + t[..., None] * d[None]  # (n, m, 3)
    dist = np.linalg.norm(p[:, None] - q, axis=2)
    k = dist.argmin(1)
    idx = np.arange(len(p))
    return dist[idx, k], q[idx, k]


def _kabsch(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t minimising |R p + t - q|."""
    pc, qc = p.mean(0), q.mean(0)
    u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
    s = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ s @ u.T
    return r, qc - r @ pc


def _rotations(n: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.concatenate([np.eye(3)[None], Rotation.random(n - 1, random_state=0).as_matrix()])


def align_to_wireframe(points_nm: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """`points_nm` rigidly moved onto the target polyhedron's edges (the target's frame), and each point's
    distance to the nearest edge."""
    verts, edges = polyhedron(spec["type"], float(spec["edge_nm"]))
    tube = float(spec.get("tube_nm", TUBE_NM))
    a = verts[[i for i, _ in edges]]
    b = verts[[j for _, j in edges]]
    pts = np.asarray(points_nm, float)
    pts = pts - pts.mean(0)

    # ICP from many starts; the best fit is the one with the most points inside the tubes, then the closest
    best = None
    for r0 in _rotations(ICP_STARTS):
        x = pts @ r0.T
        for _ in range(ICP_ITERS):
            dist, q = _closest_on_edges(x, a, b)
            r, t = _kabsch(x, q)
            x_new = x @ r.T + t
            if np.abs(x_new - x).max() < 1e-4:
                x = x_new
                break
            x = x_new
        dist, _ = _closest_on_edges(x, a, b)
        key = ((dist <= tube).mean(), -float(np.sqrt((dist ** 2).mean())))
        if best is None or key > best[0]:
            best = (key, x, dist)
    return best[1], best[2]


def fit_wireframe(points_nm: np.ndarray, spec: dict[str, Any]) -> dict[str, Any]:
    """Fit `points_nm` (n, 3) onto the target polyhedron in `spec` ({type, edge_nm[, tube_nm]})."""
    verts, edges = polyhedron(spec["type"], float(spec["edge_nm"]))
    tube = float(spec.get("tube_nm", TUBE_NM))
    x, dist = align_to_wireframe(points_nm, spec)

    precision = float((dist <= tube).mean())
    edge_recall = []
    for (i, j) in edges:
        length = float(np.linalg.norm(verts[j] - verts[i]))
        n = max(2, int(math.ceil((length - 2 * END_NM) / SAMPLE_NM)) + 1)
        s = verts[i] + np.linspace(END_NM / length, 1 - END_NM / length, n)[:, None] * (verts[j] - verts[i])
        near = np.linalg.norm(s[:, None] - x[None], axis=2).min(1)
        edge_recall.append(float((near <= tube).mean()))
    recall = float(np.mean(edge_recall))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "type": spec["type"],
        "edge_nm": float(spec["edge_nm"]),
        "tube_nm": tube,
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "edge_recall": [round(e, 3) for e in edge_recall],
        "rms_distance_nm": round(float(np.sqrt((dist ** 2).mean())), 3),
    }


def design_points_nm(design) -> np.ndarray:
    """Nucleotide positions (nm) of a scadnano design's idealised geometry, from its oxDNA export."""
    dat, _ = design.to_oxdna_format(warn_duplicate_strand_names=False)
    rows = np.array([[float(x) for x in ln.split()[:3]] for ln in dat.splitlines()[3:] if ln.strip()])
    return rows * 0.8518
