import random, math, pickle

S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
RES = {(0, 1): (3, 13), (1, 2): (6, 17), (2, 3): (10, 20),
       (3, 4): (3, 13), (4, 5): (6, 17), (5, 0): (10, 20)}

cand = {}
for pair in PAIRS:
    a, b = pair
    lo = max(S[a], S[b]) + 1
    hi = min(E[a], E[b]) - 1
    cand[pair] = [p for p in range(lo, hi + 1) if p % 21 in RES[pair]]


def cost(X):
    c = 0
    B = [set() for _ in range(6)]
    for pair in PAIRS:
        for p in X[pair]:
            B[pair[0]].add(p)
            B[pair[1]].add(p)
    for pair in PAIRS:
        a, b = pair
        pts = sorted(X[pair])
        # spacing / spread
        if not pts:
            c += 50
            continue
        gaps = [pts[0] - max(S[a], S[b])] + [pts[i + 1] - pts[i] for i in range(len(pts) - 1)] + \
               [min(E[a], E[b]) - pts[-1]]
        for g in gaps[1:-1]:
            if g < 14:
                c += 3 * (14 - g)
            if g > 42:
                c += g - 42
        for g in (gaps[0], gaps[-1]):
            if g > 30:
                c += g - 30
        for p in pts:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            if L < 18:
                c += 2 * (18 - L)
            if L > 60:
                c += L - 60
    for h in range(6):
        last = max(B[h]) if B[h] else S[h]
        e = E[h] - last
        if e < 18:
            c += 2 * (18 - e)
        if e > 60:
            c += e - 60
    return c


random.seed(3)
best_overall = None
for restart in range(20):
    X = {p: [q for q in cand[p] if random.random() < 0.3] for p in PAIRS}
    cur = cost(X)
    T = 2.0
    for step in range(40000):
        pair = random.choice(PAIRS)
        p = random.choice(cand[pair])
        Y = dict(X)
        if p in X[pair]:
            Y[pair] = [q for q in X[pair] if q != p]
        else:
            Y[pair] = sorted(X[pair] + [p])
        nc = cost(Y)
        if nc <= cur or random.random() < math.exp(-(nc - cur) / T):
            X, cur = Y, nc
        T = max(0.05, T * 0.9998)
        if cur == 0:
            break
    print("restart", restart, "cost", cur, flush=True)
    if cur == 0:
        best_overall = X
        break
if best_overall:
    for p in PAIRS:
        print(p, best_overall[p])
    pickle.dump(best_overall, open("designs/anneal_sol.pkl", "wb"))
else:
    print("none")
