import itertools, sys
sys.setrecursionlimit(10000)

RES = {(0, 1): {3, 13}, (1, 2): {6, 17}, (2, 3): {10, 20},
       (3, 4): {3, 13}, (4, 5): {6, 17}, (5, 0): {10, 20}}
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]


def valid(pair, pos):
    return pos % 21 in RES[pair]


def tile(start, end):
    """pair-only U tiling; returns list of units or None"""
    N = 6
    seen = set()

    def dfs(front, acc):
        if all(front[h] == end[h] for h in range(N)):
            return acc
        key = tuple(front)
        if key in seen:
            return None
        seen.add(key)
        m = min(front[h] for h in range(N) if front[h] < end[h])
        hmins = {h for h in range(N) if front[h] == m}
        for (a, b) in PAIRS:
            if not (hmins & {a, b}) or front[a] != front[b]:
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
                r = dfs(nf, acc + [(a, b, x, y)])
                if r is not None:
                    return r
        # single-helix staples, only at helix ends (no crossover) to resolve start/end mismatches
        for h in sorted(hmins):
            x = front[h]
            for L in range(18, 31):
                y = x + L
                if y > end[h]:
                    break
                if not (x == start[h] or (y == end[h] or end[h] - y >= 18)):
                    continue
                if x != start[h] and not (y == end[h] or end[h] - y >= 18):
                    continue
                if x != start[h]:
                    continue  # singles only from the very start of a helix
                nf = list(front)
                nf[h] = y
                r = dfs(nf, acc + [("single", h, x, y)])
                if r is not None:
                    return r
        return None

    return dfs(list(start), [])


results = []
for R01, L12, R23, L34, R45 in itertools.product(
        [113, 134, 155], [1, 22, 43], [120, 141, 162], [8, 29, 50], [106, 127, 148]):
    l1 = R01 - L12
    l2 = R23 - L12
    l3 = R23 - L34
    l4 = R45 - L34
    if not all(115 <= l <= 135 for l in (l1, l2, l3, l4)):
        continue
    # helix 0 and 5 free start: choose length 126 -> start = R01-126 (must be >=0)
    for S0 in range(0, 40):
        for S5 in range(0, 40):
            l0 = R01 - S0
            l5 = R45 - S5
            if not (115 <= l0 <= 135 and 115 <= l5 <= 135):
                continue
            start = [S0, L12, L12, L34, L34, S5]
            end = [R01, R01, R23, R23, R45, R45]
            tot = sum(end[h] - start[h] for h in range(6))
            if not 700 <= tot <= 820:
                continue
            units = tile(start, end)
            if units:
                print("span", start, end, "lengths", [end[h]-start[h] for h in range(6)], flush=True)
                results.append((start, end, units))
print("found", len(results))
import pickle
pickle.dump(results, open("designs/spans.pkl", "wb"))
