import itertools, pickle

S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
RES = {(0, 1): (3, 13), (1, 2): (6, 17), (2, 3): (10, 20),
       (3, 4): (3, 13), (4, 5): (6, 17), (5, 0): (10, 20)}


def crossover_points(pair, r, skip, phase):
    a, b = pair
    lo = max(S[a], S[b]) + 1
    hi = min(E[a], E[b]) - 1
    pts = [p for p in range(lo, hi + 1) if p % 21 == r]
    return pts[phase::skip]


opts = {p: [(r, s, ph) for r in RES[p] for s in (1, 2) for ph in (0, 1)] for p in PAIRS}
sols = []
for choice in itertools.product(*[opts[p] for p in PAIRS]):
    B = [set() for _ in range(6)]
    X = {}
    for pair, (r, s, ph) in zip(PAIRS, choice):
        pts = crossover_points(pair, r, s, ph)
        X[pair] = pts
        for p in pts:
            B[pair[0]].add(p)
            B[pair[1]].add(p)
    ok = True
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            if not 18 <= L <= 60:
                ok = False
    ends = [E[h] - (max(B[h]) if B[h] else S[h]) for h in range(6)]
    if not all(18 <= e <= 60 for e in ends):
        ok = False
    # every pair must have crossovers spread along the helix (gap <= 45)
    for pair in PAIRS:
        pts = sorted(X[pair])
        if not pts:
            ok = False
            continue
        gaps = [pts[0] - S[pair[0]]] + [pts[i + 1] - pts[i] for i in range(len(pts) - 1)] + [min(E[pair[0]], E[pair[1]]) - pts[-1]]
        if max(gaps) > 45:
            ok = False
    if ok:
        sols.append((choice, X))

print("num solutions", len(sols))
for choice, X in sols[:5]:
    print(choice, {k: len(v) for k, v in X.items()})
pickle.dump(sols, open("designs/sols3.pkl", "wb"))
