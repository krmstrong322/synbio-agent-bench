import random, itertools, math

def adj(a, b):
    (h1, v1), (h2, v2) = a, b
    if v1 == v2 and abs(h1 - h2) == 1:
        return True
    if h1 == h2 and abs(v1 - v2) == 1:
        lo = v1 if v1 < v2 else v2
        return (h1 + lo) % 2 == 1  # odd (h+v) cell pairs with the cell above
    return False

region = [(h, v) for h in range(-1, 6) for v in range(-1, 5)]
random.seed(1)
best = None
for trial in range(200000):
    start = random.choice(region)
    cells = {start}
    while len(cells) < 12:
        cand = [c for c in region if c not in cells and any(adj(c, d) for d in cells)]
        cells.add(random.choice(cand))
    cells = sorted(cells)
    ok = all(sum(adj(a, b) for b in cells if b != a) >= 2 for a in cells)
    if not ok:
        continue
    # compactness: number of adjacent pairs (more = better), then bbox
    npairs = sum(adj(a, b) for a, b in itertools.combinations(cells, 2))
    hs = [c[0] for c in cells]; vs = [c[1] for c in cells]
    score = (npairs, -(max(hs) - min(hs)) - (max(vs) - min(vs)))
    if best is None or score > best[0]:
        best = (score, cells)

score, cells = best
print("score", score)
print("cells", cells)
for c in cells:
    print(c, [d for d in cells if adj(c, d)])

# Hamiltonian path (scaffold order)
n = len(cells)
def dfs(path, used):
    if len(path) == n:
        return path
    for d in cells:
        if d not in used and adj(path[-1], d):
            r = dfs(path + [d], used | {d})
            if r:
                return r
    return None
paths = None
for s in cells:
    paths = dfs([s], {s})
    if paths:
        break
print("hamiltonian", paths)
