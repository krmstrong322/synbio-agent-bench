import random, pickle

S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
RES = {(0, 1): (3, 13), (1, 2): (6, 17), (2, 3): (10, 20),
       (3, 4): (3, 13), (4, 5): (6, 17), (5, 0): (10, 20)}


def check(X, N):
    # B[h]: dict pos -> type ('C', pair) or 'N'
    B = [dict() for _ in range(6)]
    for pair in PAIRS:
        for p in X[pair]:
            B[pair[0]][p] = ('C', pair)
            B[pair[1]][p] = ('C', pair)
    for h in range(6):
        for p in N[h]:
            if p in B[h]:
                return False
            B[h][p] = 'N'
    # pieces: for each helix, list of (start, end, boundary-type at end)
    pieces = [dict() for _ in range(6)]
    for h in range(6):
        bs = sorted(B[h])
        prev = S[h]
        for p in bs:
            pieces[h][p] = (p - prev, B[h][p])
            prev = p
        last = prev
        if not 18 <= E[h] - last <= 60:
            return False
    # standalone nick pieces
    for h in range(6):
        for p, (L, t) in pieces[h].items():
            if t == 'N' and not 18 <= L <= 60:
                return False
    # crossover staples
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            La = pieces[a][p][0]
            Lb = pieces[b][p][0]
            if not 18 <= La + Lb <= 60:
                return False
    return True


random.seed(7)
found = []
cand = {}
for pair in PAIRS:
    a, b = pair
    lo = max(S[a], S[b]) + 1
    hi = min(E[a], E[b]) - 1
    cand[pair] = {r: [p for p in range(lo, hi + 1) if p % 21 == r] for r in RES[pair]}

for it in range(2000000):
    X = {}
    for pair in PAIRS:
        r = random.choice(RES[pair])
        pts = cand[pair][r]
        # random subset with gaps in [14, 45]
        sel = []
        for p in pts:
            if not sel or (p - sel[-1] >= 14 and random.random() < 0.8):
                sel.append(p)
        X[pair] = sel
    N = [set() for _ in range(6)]
    for h in range(6):
        k = random.choice([0, 0, 1, 1, 2])
        for _ in range(k):
            N[h].add(random.randint(S[h] + 18, E[h] - 18))
    if check(X, N):
        found.append((X, N))
        if len(found) >= 3:
            break
print("found", len(found), "iters", it)
if found:
    X, N = found[0]
    for pair in PAIRS:
        print("C", pair, X[pair])
    for h in range(6):
        print("N", h, sorted(N[h]))
pickle.dump(found, open("designs/sols5.pkl", "wb"))
