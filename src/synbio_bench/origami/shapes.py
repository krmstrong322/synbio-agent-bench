"""Target shapes for flat (single-layer) origami and shape-fidelity scoring.

Shapes are polygons with unit area. Fidelity is the intersection-over-union
(IoU) between a measured 2D mask and the target polygon, scaled to the mask's
own area and placed at the best rotation, reflection and translation. So the
score measures shape, not size; scaffold length and coverage pin the size down.

Two masks are scored:
- design footprint: every designed base pair drawn as a 0.34 nm x PITCH_NM cell
  in (offset, helix row) coordinates. Cheap and visible to the agent.
- simulated outline: the relaxed oxDNA mean structure projected onto its best-
  fit plane and dilated. Measures whether the shape survives the physics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scadnano as sc

RISE_NM = 0.34  # length of one base pair along a helix
PITCH_NM = 2.6  # centre-to-centre spacing of neighbouring helices in a flat sheet
PX_NM = 1.0  # raster resolution
SIM_DILATION_NM = 1.6  # fills the gaps between projected nucleotides and helices

SHAPE_TYPES = ("rectangle", "triangle", "disk", "star", "hexagon")


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _regular(n: int, phase: float = 90.0) -> np.ndarray:
    a = np.radians(phase + 360.0 * np.arange(n) / n)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def shape_polygon(spec: dict[str, Any]) -> np.ndarray:
    """Unit-area polygon centred on its centroid, from a spec like {type: triangle}."""
    kind = spec["type"]
    if kind == "rectangle":
        a = float(spec.get("aspect", 1.0))  # width / height
        p = np.array([[-a, -1], [a, -1], [a, 1], [-a, 1]], dtype=float) / 2
    elif kind == "triangle":
        p = _regular(3)
    elif kind == "hexagon":
        p = _regular(6, phase=0)
    elif kind == "disk":
        p = _regular(128)
    elif kind == "star":
        k, inner = int(spec.get("points", 5)), float(spec.get("inner_ratio", 0.5))
        outer = _regular(k)
        mid = _regular(k, phase=90 + 180 / k) * inner
        p = np.empty((2 * k, 2))
        p[0::2], p[1::2] = outer, mid
    else:
        raise ValueError(f"unknown shape type {kind!r}; choose from {SHAPE_TYPES}")
    p = p - _poly_centroid(p)
    return p / np.sqrt(polygon_area(p))


def rotational_order(spec: dict[str, Any]) -> int:
    """n-fold rotational symmetry of a shape (all supported shapes are also mirror-symmetric)."""
    return {"rectangle": 2, "triangle": 3, "hexagon": 6, "disk": 360, "star": int(spec.get("points", 5))}[spec["type"]]


def _poly_centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * a)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over points."""
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    crosses = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
    return crosses.sum(1) % 2 == 1


# ---------------------------------------------------------------------------
# Masks and IoU
# ---------------------------------------------------------------------------


def best_iou(cells: np.ndarray, shape: dict[str, Any], angle_step: float = 2.0) -> dict[str, float]:
    """IoU between filled pixel centres `cells` (N, 2; PX_NM grid) and the best-placed target polygon.

    Searches rotations over one symmetry period, then a +-2 px translation. All
    supported shapes are mirror-symmetric, so reflections need no search.
    """
    poly_unit = shape_polygon(shape)
    if len(cells) == 0:
        return {"iou": 0.0, "angle_deg": 0.0}
    area = len(cells) * PX_NM**2
    centre = cells.mean(0)
    poly0 = poly_unit * np.sqrt(area)
    reach = np.abs(poly0).max() + 3 * PX_NM
    # candidate pixels: the cells' bounding box padded by the polygon's reach, on the cells' own pixel lattice
    origin = cells.min(0)
    lo_i = np.floor((np.minimum(origin, centre - reach) - origin) / PX_NM).astype(int) - 3
    hi_i = np.ceil((np.maximum(cells.max(0), centre + reach) - origin) / PX_NM).astype(int) + 3
    gx = origin[0] + PX_NM * np.arange(lo_i[0], hi_i[0] + 1)
    gy = origin[1] + PX_NM * np.arange(lo_i[1], hi_i[1] + 1)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    filled = np.zeros(len(grid), bool)
    idx = np.rint((cells - [gx[0], gy[0]]) / PX_NM).astype(int)
    filled[idx[:, 1] * len(gx) + idx[:, 0]] = True

    def iou(theta: float, shift: np.ndarray) -> float:
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        p = poly0 @ np.array([[c, s], [-s, c]]) + centre + shift
        inside = points_in_polygon(grid, p)
        inter = np.count_nonzero(inside & filled)
        return inter / (np.count_nonzero(inside) + len(cells) - inter)

    period = 360.0 / rotational_order(shape)
    best_val, best_t = max((iou(t, np.zeros(2)), t) for t in np.arange(0, period, min(angle_step, period)))
    best_shift = np.zeros(2)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            v = iou(best_t, np.array([dx, dy]) * PX_NM)
            if v > best_val:
                best_val, best_shift = v, np.array([dx, dy]) * PX_NM
    c, s = np.cos(np.radians(best_t)), np.sin(np.radians(best_t))
    placed = poly0 @ np.array([[c, s], [-s, c]]) + centre + best_shift
    return {"iou": round(float(best_val), 4), "angle_deg": float(best_t), "polygon": placed.round(2).tolist()}


def footprint_cells(design: sc.Design) -> np.ndarray | None:
    """Pixel centres covered by designed base pairs, or None if the helices aren't in one flat row."""
    helices = [h for h in design.helices.values() if h.grid_position is not None]
    if not helices or len(helices) != len(design.helices):
        return None
    hs, vs = {h.grid_position[0] for h in helices}, {h.grid_position[1] for h in helices}
    if len(hs) == 1:
        row_of = {h.idx: h.grid_position[1] for h in helices}
    elif len(vs) == 1:
        row_of = {h.idx: h.grid_position[0] for h in helices}
    else:
        return None
    paired: dict[int, set[int]] = {}
    for strand in design.strands:
        for d in strand.domains:
            if isinstance(d, sc.Domain):
                key = (d.helix, d.forward)
                paired.setdefault(key, set()).update(range(d.start, d.end))  # type: ignore[arg-type]
    cells = set()
    for h in design.helices:
        offsets = sorted(paired.get((h, True), set()) & paired.get((h, False), set()))
        rows = _pixels(row_of[h] * PITCH_NM, (row_of[h] + 1) * PITCH_NM)
        for a, b in _runs(offsets):
            for i in _pixels(a * RISE_NM, b * RISE_NM):
                cells.update((i, j) for j in rows)
    return (np.array(sorted(cells), dtype=float).reshape(-1, 2) + 0.5) * PX_NM


def _pixels(lo: float, hi: float) -> range:
    """Indices of pixels whose centres ((i + 0.5) * PX_NM) fall in [lo, hi)."""
    return range(int(np.ceil(lo / PX_NM - 0.5)), int(np.ceil(hi / PX_NM - 0.5)))


def _runs(offsets: list[int]) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs in a sorted list of offsets."""
    runs: list[tuple[int, int]] = []
    for o in offsets:
        if runs and runs[-1][1] == o:
            runs[-1] = (runs[-1][0], o + 1)
        else:
            runs.append((o, o + 1))
    return runs


def project_to_plane(points_nm: np.ndarray) -> np.ndarray:
    """Coordinates along the structure's principal axes (largest first); columns 0-1 span its best-fit plane."""
    c = points_nm - points_nm.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt.T


def outline_cells(points_nm: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a 3D structure onto its best-fit plane and rasterise with dilation.

    Returns (cells, flatness) where flatness = extent along the plane normal /
    largest in-plane extent (≈0 for a flat sheet).
    """
    proj = project_to_plane(points_nm)
    xy, z = proj[:, :2], proj[:, 2]
    flatness = float((z.max() - z.min()) / max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9))
    pix = np.unique(np.round(xy / PX_NM).astype(int), axis=0)
    r = int(np.ceil(SIM_DILATION_NM / PX_NM))
    kernel = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    dil = np.unique(np.concatenate([pix + k for k in kernel]), axis=0)
    return dil.astype(float) * PX_NM, flatness


def design_shape_iou(design: sc.Design, shape: dict[str, Any]) -> dict[str, Any]:
    cells = footprint_cells(design)
    if cells is None:
        return {"iou": 0.0, "error": "helices are not in a single flat row of the lattice"}
    fit = best_iou(cells, shape)
    fit.pop("polygon")  # only figures need it; keep tool output short
    return fit


def simulated_shape_iou(mean_structure_nm: np.ndarray, shape: dict[str, Any]) -> dict[str, Any]:
    cells, flatness = outline_cells(mean_structure_nm)
    out = best_iou(cells, shape)
    out.pop("polygon")
    out["flatness"] = round(flatness, 3)
    return out
