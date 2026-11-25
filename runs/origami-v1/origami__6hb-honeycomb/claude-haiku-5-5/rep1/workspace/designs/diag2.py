import itertools
S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
RES = {(0, 1): (3, 13), (1, 2): (6, 17), (2, 3): (10, 20),
       (3, 4): (3, 13), (4, 5): (6, 17), (5, 0): (10, 20)}
best = None
for rs in itertools.product(*[RES[p] for p in PAIRS]):
    X = {}
    for pair, r in zip(PAIRS, rs):
        a, b = pair
        lo = max(S[a], S[b]) + 1
        hi = min(E[a], E[b]) - 1
        X[pair] = [p for p in range(lo, hi + 1) if p % 21 == r]
    B = [set() for _ in range(6)]
    for pair in PAIRS:
        for p in X[pair]:
            B[pair[0]].add(p); B[pair[1]].add(p)
    bad = []
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            if not 18 <= L <= 60:
                bad.append(("staple", pair, p, L))
    for h in range(6):
        last = max(B[h]) if B[h] else S[h]
        if not 18 <= E[h] - last <= 60:
            bad.append(("end", h, last, E[h] - last))
    print(rs, len(bad), bad)
