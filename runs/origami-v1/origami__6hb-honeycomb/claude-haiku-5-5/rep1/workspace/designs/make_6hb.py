import sys
sys.setrecursionlimit(10000)
import scadnano as sc

# Ring of 6 honeycomb helices (consecutive entries adjacent, last adjacent to first)
POS = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]
N = 6

# Scaffold span per helix (fwd for even helices, rev for odd); junctions verified
# with find_crossover_offsets (positions valid mod 21).
SCAF = {
    0: (29, 155, True),
    1: (22, 155, False),
    2: (22, 141, True),
    3: (8, 141, False),
    4: (8, 127, True),
    5: (1, 127, False),
}
# valid crossover residues (mod 21) for ring pair (a, b), a = lower index in PAIRS
RES = {
    (0, 1): {3, 13},
    (1, 2): {6, 17},
    (2, 3): {10, 20},
    (3, 4): {3, 13},
    (4, 5): {6, 17},
    (5, 0): {10, 20},
}
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
start = [SCAF[h][0] for h in range(N)]
end = [SCAF[h][1] for h in range(N)]


def valid(pair, pos):
    return pos % 21 in RES[pair]


def staple_fwd(h):
    return not SCAF[h][2]


seen = set()


def dfs(front, acc):
    if all(front[h] == end[h] for h in range(N)):
        return acc
    key = tuple(front)
    if key in seen:
        return None
    seen.add(key)
    # always work on the helix with the smallest unfinished frontier
    hmin = min((h for h in range(N) if front[h] < end[h]), key=lambda h: front[h])
    # pair U-staples involving hmin: both helices start at the same frontier x
    for (a, b) in PAIRS:
        if hmin not in (a, b):
            continue
        if front[a] != front[b]:
            continue
        x = front[a]
        lim = min(end[a], end[b])
        cands = []
        for L in range(9, 31):
            y = x + L
            if y > lim:
                break
            if valid((a, b), y) or valid((a, b), x):
                cands.append((abs(L - 21), y))
        cands.sort()
        for _, y in cands:
            nf = list(front)
            nf[a] = y
            nf[b] = y
            r = dfs(nf, acc + [("pair", a, b, x, y)])
            if r is not None:
                return r
    # single-helix staples (no crossover), used only where needed
    for h in [hmin]:
        x = front[h]
        if x >= end[h]:
            continue
        for L in range(18, 31):
            y = x + L
            if y > end[h]:
                break
            if not (y == end[h] or end[h] - y >= 18):
                continue
            nf = list(front)
            nf[h] = y
            r = dfs(nf, acc + [("single", h, x, y)])
            if r is not None:
                return r
    return None


sol = dfs(start, [])
assert sol is not None, "no staple tiling found"
print("units:", len(sol))

design_helices = [sc.Helix(max_offset=160, grid_position=POS[i]) for i in range(N)]
design = sc.Design(helices=design_helices, strands=[], grid=sc.honeycomb)

dom = []
for h in range(N):
    s, e, fwd = SCAF[h]
    dom.append(sc.Domain(helix=h, forward=fwd, start=s, end=e))
design.add_strand(sc.Strand(domains=dom, is_scaffold=True))

for unit in sol:
    if unit[0] == "single":
        _, h, x, y = unit
        design.add_strand(sc.Strand(domains=[sc.Domain(helix=h, forward=staple_fwd(h), start=x, end=y)]))
        continue
    _, a, b, x, y = unit
    fa = staple_fwd(a)
    da = sc.Domain(helix=a, forward=fa, start=x, end=y)
    db = sc.Domain(helix=b, forward=not fa, start=x, end=y)
    fwd_d, rev_d = (da, db) if fa else (db, da)
    if valid((a, b), y):
        doms = [fwd_d, rev_d]      # crossover at y
    else:
        doms = [rev_d, fwd_d]      # crossover at x
    design.add_strand(sc.Strand(domains=doms))

design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/6hb.sc")

scaf = [s for s in design.strands if s.is_scaffold][0]
print("scaffold length", scaf.dna_length())
lens = sorted(s.dna_length() for s in design.strands if not s.is_scaffold)
print("staples", len(lens), "min", lens[0], "max", lens[-1])
for u in sol:
    pass
