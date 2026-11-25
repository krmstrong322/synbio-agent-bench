import itertools, pickle

S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
RES = {(0, 1): (3, 13), (1, 2): (6, 17), (2, 3): (10, 20),
       (3, 4): (3, 13), (4, 5): (6, 17), (5, 0): (10, 20)}


def crossover_points(pair, r, skip):
    a, b = pair
    lo = max(S[a], S[b]) + 1
    hi = min(E[a], E[b]) - 1
    pts = [p for p in range(lo, hi + 1) if p % 21 == r]
    return pts[::skip]


sols = []
for choice in itertools.product(*[[(r, s) for r in RES[p] for s in (1, 2)] for p in PAIRS]):
    B = [set() for _ in range(6)]
    X = {}
    for pair, (r, s) in zip(PAIRS, choice):
        pts = crossover_points(pair, r, s)
        X[pair] = pts
        for p in pts:
            B[pair[0]].add(p)
            B[pair[1]].add(p)
    ok = True
    lens = {}
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            lens[(pair, p)] = L
            if not 18 <= L <= 60:
                ok = False
    endpc = []
    for h in range(6):
        last = max(B[h]) if B[h] else S[h]
        e = E[h] - last
        endpc.append(e)
        if not 18 <= e <= 60:
            ok = False
    if ok:
        sols.append((choice, X, endpc))

print("num solutions", len(sols))
for choice, X, endpc in sols[:10]:
    print(choice, {k: len(v) for k, v in X.items()}, endpc)
pickle.dump(sols, open("designs/sols2.pkl", "wb"))
