import random, pickle

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


def check(X):
    B = [set() for _ in range(6)]
    for pair in PAIRS:
        for p in X[pair]:
            B[pair[0]].add(p)
            B[pair[1]].add(p)
    for pair in PAIRS:
        a, b = pair
        for p in X[pair]:
            pa = max([q for q in B[a] | {S[a]} if q < p])
            pb = max([q for q in B[b] | {S[b]} if q < p])
            L = (p - pa) + (p - pb)
            if not 18 <= L <= 60:
                return False
    for h in range(6):
        last = max(B[h]) if B[h] else S[h]
        if not 18 <= E[h] - last <= 60:
            return False
    for pair in PAIRS:
        pts = sorted(X[pair])
        if len(pts) < 2:
            return False
        a, b = pair
        gaps = [pts[0] - max(S[a], S[b])] + [pts[i + 1] - pts[i] for i in range(len(pts) - 1)] + \
               [min(E[a], E[b]) - pts[-1]]
        if max(gaps) > 45 or min(gaps[1:-1]) < 14:
            return False
    return True


random.seed(1)
found = []
for it in range(400000):
    X = {}
    for pair in PAIRS:
        X[pair] = [p for p in cand[pair] if random.random() < 0.5]
    # enforce spacing >=14 within a pair
    good = True
    for pair in PAIRS:
        pts = X[pair]
        for i in range(len(pts) - 1):
            if pts[i + 1] - pts[i] < 14:
                good = False
                break
        if not good:
            break
    if not good:
        continue
    if check(X):
        found.append(X)
        if len(found) >= 5:
            break
print("found", len(found))
if found:
    X = found[0]
    for pair in PAIRS:
        print(pair, X[pair])
pickle.dump(found, open("designs/sols4.pkl", "wb"))
