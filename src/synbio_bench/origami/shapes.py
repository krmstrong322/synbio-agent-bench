"""Target shapes for flat (single-layer) origami and shape-fidelity scoring.

Shapes are polygons with unit area. Fidelity is the intersection-over-union
(IoU) between a measured 2D mask and the target polygon, scaled to the mask's
own area and placed at the best rotation, reflection and translation. So the
score measures shape, not size; scaffold length and coverage pin the size down.

Three masks are scored:
- design footprint: every designed base pair drawn as a 0.34 nm x PITCH_NM cell
  in (offset, helix row) coordinates. Cheap and visible to the agent.
- unrolled sheet (graded): the relaxed oxDNA mean structure flattened along its
  own surface, so rolling or bending in solution doesn't count, but tearing or
  splitting does.
- projected outline: the mean structure projected onto its best-fit plane, with
  flatness. Describes how the sheet sits in solution (sheets twist and can roll up).
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
    cells = fill_holes(cells)
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


def fill_holes(cells: np.ndarray) -> np.ndarray:
    """Add the pixels enclosed by a mask, so a shape is judged by its outline (a hollow triangle counts as a triangle)."""
    from scipy.ndimage import binary_fill_holes

    origin = cells.min(0)
    idx = np.rint((cells - origin) / PX_NM).astype(int) + 1
    img = np.zeros(idx.max(0) + 2, bool)
    img[idx[:, 0], idx[:, 1]] = True
    filled = np.argwhere(binary_fill_holes(img))
    return (filled - 1) * PX_NM + origin


def helix_frames(design: sc.Design) -> dict[int, tuple[np.ndarray, np.ndarray, float, float]]:
    """Each helix's axis in 3D, as scadnano exports it: {helix: (origin_nm, unit direction, rise_nm, spacing_nm)}."""
    from scadnano.scadnano import _oxdna_get_helix_vectors

    out = {}
    for idx, h in design.helices.items():
        origin, forward, _ = _oxdna_get_helix_vectors(design, h)
        geometry = design.groups[h.group].geometry or design.geometry
        out[idx] = (np.array([origin.x, origin.y, origin.z]) * OX_TO_NM_EXPORT, np.array([forward.x, forward.y, forward.z]),
                    geometry.rise_per_base_pair, geometry.distance_between_helices())
    return out


OX_TO_NM_EXPORT = 0.8518  # scadnano writes positions in oxDNA length units


def planar_sheet(design: sc.Design, tol_nm: float = 1.0) -> bool:
    """True if every helix axis lies in one plane, i.e. the design is a single flat layer.

    Works for one lattice row and for several helix groups rotated within the sheet's plane.
    """
    frames = helix_frames(design)
    if not frames:
        return False
    pts = []
    for h, (o, f, rise, _) in frames.items():
        hx = design.helices[h]
        pts += [o + f * hx.min_offset * rise, o + f * hx.max_offset * rise]
    pts = np.array(pts)
    c = pts - pts.mean(0)
    normal = np.linalg.svd(c, full_matrices=False)[2][-1]
    return bool(np.abs(c @ normal).max() <= tol_nm)


def _paired_runs(design: sc.Design) -> dict[int, list[tuple[int, int]]]:
    paired: dict[tuple[int, bool], set[int]] = {}
    for strand in design.strands:
        for d in strand.domains:
            if isinstance(d, sc.Domain):
                paired.setdefault((d.helix, d.forward), set()).update(range(d.start, d.end))  # type: ignore[arg-type]
    return {h: _runs(sorted(paired.get((h, True), set()) & paired.get((h, False), set()))) for h in design.helices}


def _group_footprint_cells(design: sc.Design) -> np.ndarray | None:
    """Footprint of a design made of several helix groups: each paired run drawn as a band one helix spacing wide
    around its real axis, in the sheet's plane."""
    if not planar_sheet(design):
        return None
    frames = helix_frames(design)
    allpts = np.array([o for o, *_ in frames.values()] + [o + f * 50 for o, f, *_ in frames.values()])
    centre = allpts.mean(0)
    basis = np.linalg.svd(allpts - centre, full_matrices=False)[2][:2]
    step = PX_NM / 4
    cells = set()
    for h, runs in _paired_runs(design).items():
        o, f, rise, spacing = frames[h]
        f2 = basis @ f
        f2 /= np.linalg.norm(f2)
        side = np.array([-f2[1], f2[0]])
        o2 = basis @ (o - centre)
        across = np.arange(-spacing / 2 + step / 2, spacing / 2, step)
        for a, b in runs:
            along = np.arange(a * rise + step / 2, b * rise, step)
            pts = o2 + along[:, None, None] * f2 + across[None, :, None] * side
            cells.update(map(tuple, np.floor(pts.reshape(-1, 2) / PX_NM).astype(int)))
    return (np.array(sorted(cells), dtype=float).reshape(-1, 2) + 0.5) * PX_NM


def footprint_cells(design: sc.Design) -> np.ndarray | None:
    """Pixel centres covered by designed base pairs, or None if the design isn't one flat layer.

    One lattice row is drawn in (offset, row) coordinates with RISE_NM x PITCH_NM cells. Designs made of
    several helix groups (e.g. trapezoids rotated in the sheet's plane) are drawn from their real helix axes.
    """
    if len({h.group for h in design.helices.values()}) > 1:
        return _group_footprint_cells(design)
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
    return _dilated_cells(xy), flatness


def _dilated_cells(xy: np.ndarray) -> np.ndarray:
    """Pixel centres within SIM_DILATION_NM of any 2D point."""
    pix = np.unique(np.round(xy / PX_NM).astype(int), axis=0)
    r = int(np.ceil(SIM_DILATION_NM / PX_NM))
    kernel = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    dil = np.unique(np.concatenate([pix + k for k in kernel]), axis=0)
    return dil.astype(float) * PX_NM


def design_shape_iou(design: sc.Design, shape: dict[str, Any]) -> dict[str, Any]:
    cells = footprint_cells(design)
    if cells is None:
        return {"iou": 0.0, "error": "helices are not in a single flat layer"}
    fit = best_iou(cells, shape)
    fit.pop("polygon", None)  # only figures need it; keep tool output short
    return fit


def simulated_shape_iou(mean_structure_nm: np.ndarray, shape: dict[str, Any]) -> dict[str, Any]:
    cells, flatness = outline_cells(mean_structure_nm)
    out = best_iou(cells, shape)
    out.pop("polygon", None)
    out["flatness"] = round(flatness, 3)
    return out


# ---------------------------------------------------------------------------
# Unrolled (developed-surface) outline
# ---------------------------------------------------------------------------

UNROLL_STEP_BP = 4  # sample the helix axis every 4 bp (1.36 nm)
UNROLL_MAX_GAP_NM = 5.0  # neighbouring-helix points further apart than this count as torn
UNROLL_DIAGONAL_STEPS = 6  # diagonal links up to 6 samples (~8 nm) along the neighbouring helix


def helix_axis_points(points_nm: np.ndarray, slots: list, step: int = UNROLL_STEP_BP) -> dict[tuple[int, int], np.ndarray]:
    """Midpoints of designed base pairs every `step` offsets: {(helix, offset): xyz}."""
    index = {(s[0], s[1], s[3]): k for k, s in enumerate(slots) if s and s[2] < 0}
    return {
        (h, off): (points_nm[k] + points_nm[index[(h, off, False)]]) / 2
        for (h, off, fwd), k in index.items()
        if fwd and off % step == 0 and (h, off, False) in index
    }


def crossover_helix_pairs(slots: list, n3: np.ndarray) -> set[tuple[int, int]]:
    """Helix pairs joined by at least one crossover (a backbone bond between helices)."""
    pairs = set()
    for i, j in enumerate(n3):
        if j >= 0 and slots[i] and slots[j] and slots[i][0] != slots[j][0]:
            pairs.add(tuple(sorted((slots[i][0], slots[j][0]))))
    return pairs


def crossover_links(slots: list, n3: np.ndarray, max_loop: int = 8) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """Every backbone connection between two helices, also across short unpaired loops (e.g. bridging staples
    between domains at an angle): [((helix, offset), (helix2, offset2), unpaired bases in between)]."""
    links = []
    for i, j in enumerate(n3):
        if j < 0 or not slots[i]:
            continue
        loop = 0
        while j >= 0 and not slots[j] and loop < max_loop:
            j, loop = n3[j], loop + 1
        if j >= 0 and slots[j] and slots[j][0] != slots[i][0]:
            links.append(((slots[i][0], slots[i][1]), (slots[j][0], slots[j][1]), loop))
    return links


def unroll_sheet(
    axis: dict[tuple[int, int], np.ndarray], helix_pairs: set[tuple[int, int]], step: int = UNROLL_STEP_BP,
    links: list[tuple[tuple[int, int], tuple[int, int], int]] = (),
) -> dict[str, Any]:
    """Flatten a (possibly rolled or bent) sheet into 2D, preserving distances along its surface.

    Builds a graph over helix-axis points: edges along each helix and between
    crossover-connected helices at the same offset (only where they are still
    within UNROLL_MAX_GAP_NM, so a split seam stays split). Geodesic distances on
    that graph are embedded in 2D by classical MDS. Neighbours come from the
    design, not from proximity, so the two edges of a rolled-up tube are never joined.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path

    keys = sorted(axis)
    idx = {k: i for i, k in enumerate(keys)}
    pts = np.array([axis[k] for k in keys])
    rows, cols = [], []
    by_helix: dict[int, list[int]] = {}
    for h, off in keys:
        by_helix.setdefault(h, []).append(off)
    for h, offs in by_helix.items():  # along each helix, bridging small gaps (e.g. deletions)
        for a, b in zip(offs, offs[1:]):
            if b - a <= 3 * step:
                rows.append(idx[(h, a)])
                cols.append(idx[(h, b)])
    # Across neighbouring helices, straight and diagonal: with straight links only, path lengths
    # are city-block distances and the unrolled sheet comes out inflated along its diagonals.
    for h1, h2 in helix_pairs:
        for off in by_helix.get(h1, []):
            i = idx[(h1, off)]
            for k in range(-UNROLL_DIAGONAL_STEPS, UNROLL_DIAGONAL_STEPS + 1):
                j = idx.get((h2, off + k * step))
                along = abs(k) * step * RISE_NM
                if j is not None and np.linalg.norm(pts[i] - pts[j]) <= np.hypot(UNROLL_MAX_GAP_NM, 1.3 * along):
                    rows.append(i)
                    cols.append(j)
    # Crossovers between helices that don't run side by side (e.g. trapezoids meeting at a seam) join the
    # sheet only where they actually connect.
    for (h1, o1), (h2, o2), loop in links:
        if h1 not in by_helix or h2 not in by_helix:
            continue
        i = idx[(h1, min(by_helix[h1], key=lambda o: abs(o - o1)))]
        j = idx[(h2, min(by_helix[h2], key=lambda o: abs(o - o2)))]
        if np.linalg.norm(pts[i] - pts[j]) <= UNROLL_MAX_GAP_NM + 0.6 * loop:
            rows.append(i)
            cols.append(j)
    n = len(keys)
    w = np.linalg.norm(pts[rows] - pts[cols], axis=1) + 1e-9
    graph = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    main = np.flatnonzero(labels == np.bincount(labels).argmax())
    d = shortest_path(graph[main][:, main], directed=False)
    m = len(main)
    j = np.eye(m) - 1.0 / m
    b = -0.5 * j @ (d**2) @ j
    vals, vecs = np.linalg.eigh(b)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.maximum(vals[top], 0))
    xy = _local_stress_fit(xy, d, pts[main])
    return {"xy": xy, "keys": [keys[i] for i in main], "components": int(n_comp), "coverage": m / n}


UNROLL_LOCAL_NM = 10.0  # only points this close along the surface shape the final layout
UNROLL_FIT_ITERS = 200


def _local_stress_fit(xy: np.ndarray, geodesic: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Refine an MDS layout so that nearby points keep their true separation (weighted stress majorization).

    Graph path lengths over-state distances: they zig-zag between helices, and across a hole (a hollow
    triangle) or a notch they go around it, which classical MDS spreads over the whole layout. For points
    within UNROLL_LOCAL_NM of each other along the surface, the straight 3D distance is within ~1% of the
    surface distance even on a rolled sheet, so the layout is fitted to those alone.
    """
    w = ((geodesic > 0) & (geodesic <= UNROLL_LOCAL_NM)).astype(float)
    if not w.any():
        return xy
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    v = -w
    np.fill_diagonal(v, w.sum(1))
    v_pinv = np.linalg.pinv(v)
    x = xy.copy()
    for _ in range(UNROLL_FIT_ITERS):
        dist = np.linalg.norm(x[:, None] - x[None], axis=-1)
        ratio = np.where(dist > 1e-9, w * d / np.maximum(dist, 1e-9), 0.0)
        bmat = -ratio
        np.fill_diagonal(bmat, ratio.sum(1))
        x_new = v_pinv @ bmat @ x
        if np.abs(x_new - x).max() < 1e-3:
            return x_new
        x = x_new
    return x


DESIGN_NEIGHBOUR_NM = 3.4  # helix-axis points this close in the designed geometry are neighbours on the sheet


def designed_contacts(design_points_nm: np.ndarray, slots: list) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """Axis points on different helices that the design places side by side, e.g. across a seam between
    trapezoids: [((helix, offset), (helix2, offset2), 0)]."""
    from scipy.spatial import cKDTree

    axis = helix_axis_points(design_points_nm, slots)
    keys = list(axis)
    tree = cKDTree(np.array([axis[k] for k in keys]))
    return [(keys[i], keys[j], 0) for i, j in tree.query_pairs(DESIGN_NEIGHBOUR_NM) if keys[i][0] != keys[j][0]]


def unrolled_shape_iou(
    points_nm: np.ndarray, slots: list, n3: np.ndarray, shape: dict[str, Any], design_points_nm: np.ndarray | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """IoU of the unrolled sheet with the target. Returns (metrics, unroll data for figures).

    Points cut off from the main sheet (a torn piece) are left out, so tearing lowers the IoU.
    `design_points_nm` (the designed starting geometry) adds contacts the design places side by side
    without a crossover, such as seams between domains; they count only while they stay close.
    """
    axis = helix_axis_points(points_nm, slots)
    if len(axis) < 10:
        return {"iou": 0.0, "error": "too few paired helix-axis points to unroll"}, {}
    links = crossover_links(slots, n3)
    if design_points_nm is not None:
        links += designed_contacts(design_points_nm, slots)
    unrolled = unroll_sheet(axis, crossover_helix_pairs(slots, n3), links=links)
    fit = best_iou(_dilated_cells(unrolled["xy"]), shape)
    fit.pop("polygon", None)
    fit.update(components=unrolled["components"], coverage=round(unrolled["coverage"], 4))
    return fit, unrolled
