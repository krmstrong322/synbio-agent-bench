import math, itertools
import numpy as np
import scadnano as sc

# ---------- geometry ----------
EDGE = 35.0
S_ = EDGE / (2 * math.sqrt(2))
V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * S_
EDGES = [(i, j) for i in range(4) for j in range(i + 1, 4)]  # pair p -> helices 2p (row0), 2p+1 (row1)
L = 84
RISE = 0.34
HALF = L * RISE / 2
SPACING = 2.6
LOOP_NM = 0.6


def group_angles(forward, side):
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def side_candidates(p):
    i, j = EDGES[p]
    M = (V[i] + V[j]) / 2
    t = unit(V[j] - V[i])
    n = unit(M)
    cands = {"tan": unit(np.cross(n, t))}
    for k in range(4):
        if k in (i, j):
            continue
        d = V[k] - M
        d = d - d.dot(t) * t
        cands[f"face{k}"] = unit(d)
    return t, M, cands


def frames(sides):
    """sides: dict p -> side vector. returns P0,T,SD lists."""
    P0, T, SD = [], [], []
    for p, (i, j) in enumerate(EDGES):
        M = (V[i] + V[j]) / 2
        t = unit(V[j] - V[i])
        s = sides[p]
        P0.append(M - HALF * t - 1.3 * s)
        T.append(t)
        SD.append(s)
    return P0, T, SD


def pos_fn(P0, T, SD):
    def pos(h, off):
        p, r = h // 2, h % 2
        return P0[p] + r * SPACING * SD[p] + off * RISE * T[p]
    return pos


def end_off(h, vertex):
    i, j = EDGES[h // 2]
    return 0 if vertex == i else L


def loop_len(d):
    return max(2, math.ceil(d / LOOP_NM))


def route_search(pos, bound=56):
    def hop_nm(h1, v1, h2, v2):
        return float(np.linalg.norm(pos(h1, end_off(h1, v1)) - pos(h2, end_off(h2, v2))))
    best = [bound, None]

    def rec(path, cur_v, cost):
        if cost >= best[0]:
            return
        if len(path) == 12:
            best[0], best[1] = cost, list(path)
            return
        used = {x[0]: x for x in path}
        for h in range(12):
            if h in used:
                continue
            p = h // 2
            i, j = EDGES[p]
            partner = h ^ 1
            options = [i, j] if not path else [cur_v]
            for e in options:
                if e not in (i, j):
                    continue
                fwd = (e == i)
                if partner in used and used[partner][1] == fwd:
                    continue
                if path and path[-1][0] // 2 == p:
                    continue  # no scaffold turns inside a pair
                exitv = j if e == i else i
                c = 0
                if path:
                    prev = path[-1]
                    c = loop_len(hop_nm(prev[0], prev[3], h, e))
                rec(path + [(h, fwd, e, exitv)], exitv, cost + c)

    rec([], None, 0)
    return best, hop_nm


if __name__ == "__main__":
    # choose side per edge among candidates, minimising scaffold loop nt
    names = [[k for k in side_candidates(p)[2].keys() if k != "tan"] for p in range(6)]
    results = []
    for combo in itertools.product(*[range(len(nm)) for nm in names]):
        sides = {p: side_candidates(p)[2][names[p][combo[p]]] for p in range(6)}
        P0, T, SD = frames(sides)
        best, _ = route_search(pos_fn(P0, T, SD))
        results.append((best[0], combo))
    results.sort(key=lambda x: x[0])
    print("best costs:", [r[0] for r in results[:8]])
    print("names example:", names)
    cost, combo = results[0]
    print("chosen", cost, [names[p][combo[p]] for p in range(6)])
    with open("designs/best_combo.txt", "w") as f:
        f.write(repr((cost, [names[p][combo[p]] for p in range(6)])))
