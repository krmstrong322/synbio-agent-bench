import itertools
def nbrs(c):
    h, v = c
    vert = (h, v+1) if (h+v) % 2 == 1 else (h, v-1)
    return [(h-1, v), (h+1, v), vert]
cells = [(h, v) for h in range(0, 6) for v in range(0, 4)]
best = []
# grow connected sets, check min degree >= 2 and count edges
def degree_ok(S):
    return all(sum(1 for n in nbrs(c) if n in S) >= 2 for c in S)
seen = set()
import random
random.seed(1)
results = []
for trial in range(200000):
    start = random.choice(cells)
    S = {start}
    while len(S) < 12:
        frontier = set()
        for c in S:
            for n in nbrs(c):
                if n in cells and n not in S:
                    frontier.add(n)
        S.add(random.choice(sorted(frontier)))
    key = frozenset(S)
    if key in seen:
        continue
    seen.add(key)
    if degree_ok(S):
        edges = sum(1 for c in S for n in nbrs(c) if n in S) // 2
        xs = [c[0] for c in S]; ys = [c[1] for c in S]
        results.append((edges, max(xs)-min(xs), max(ys)-min(ys), sorted(S)))
results.sort(key=lambda r: (-r[0], r[1]+r[2]))
for r in results[:8]:
    print(r)
