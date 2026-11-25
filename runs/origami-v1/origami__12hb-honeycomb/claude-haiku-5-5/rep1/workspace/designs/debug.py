import random
src = open("designs/build12.py").read().replace("\nmain()\n", "\n")
g = {}
exec(compile(src, "b", "exec"), g)
rng = random.Random(2)
from collections import Counter
c = Counter()
ex = None
for t in range(3000):
    allc = []
    for (a, b) in g["EDGES"]:
        for o in g["cands"](a, b):
            allc.append((o, a, b))
    allc.sort(key=lambda t: rng.random())
    X = {h: [] for h in range(12)}
    cnt = {e: 0 for e in g["EDGES"]}
    for o, a, b in allc:
        if any(abs(o - x) < 20 for x in X[a]) or any(abs(o - x) < 20 for x in X[b]):
            continue
        X[a].append(o); X[b].append(o); cnt[(a, b)] += 1
    if min(cnt.values()) < 2:
        c["edge<2"] += 1
    bad_end = False
    for h in range(12):
        if not X[h]:
            bad_end = True; continue
        if min(X[h]) - g["LO"][h] > 30 or g["HI"][h] - max(X[h]) > 30:
            bad_end = True
    if bad_end:
        c["end"] += 1
    if min(cnt.values()) >= 2 and not bad_end:
        c["ok"] += 1
        ex = X
print(c)
print(ex)
print(g["cands"](0, 1), g["cands"](2, 3))
