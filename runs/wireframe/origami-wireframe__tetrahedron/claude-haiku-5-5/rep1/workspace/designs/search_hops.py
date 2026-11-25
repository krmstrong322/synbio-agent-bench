import math, random, itertools
import numpy as np
from collections import defaultdict

s = 35.0 / (2 * math.sqrt(2))
VERTS = [s * np.array(p, float) for p in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
RISE = 0.34
SP = 2.6
edges = [(v, w) for v in range(4) for w in range(v + 1, 4)]
eidx = {e: i for i, e in enumerate(edges)}
arcs = [(a, b) for a in range(4) for b in range(4) if a != b]


def euler_rand(arcs, start):
    while True:
        out = defaultdict(list)
        al = arcs[:]
        random.shuffle(al)
        for (a, b) in al:
            out[a].append((a, b))
        stack = [(start, None)]
        circ = []
        while stack:
            v, e = stack[-1]
            if out[v]:
                arc = out[v].pop()
                stack.append((arc[1], arc))
            else:
                stack.pop()
                if e is not None:
                    circ.append(e)
        circ.reverse()
        if len(circ) == 12:
            return circ


def evaluate(N, signs, circuit):
    HALF = N * RISE / 2
    geom = {}
    for i, (v, w) in enumerate(edges):
        A, B = VERTS[v], VERTS[w]
        M = (A + B) / 2
        t = (B - A) / np.linalg.norm(B - A)
        n = M / np.linalg.norm(M)
        side = np.cross(n, t)
        side = side / np.linalg.norm(side) * signs[i]
        p0 = M - HALF * t - (SP / 2) * side
        geom[i] = (p0, t, side)

    def ap(e, row, o):
        p0, t, side = geom[e]
        return p0 + row * SP * side + o * RISE * t

    def hid(a, b):
        v, w = min(a, b), max(a, b)
        e = eidx[(v, w)]
        row = 0 if a < b else 1
        return e, row, (a < b)

    tot = 0
    ds = []
    for k in range(11):
        a1, b1 = circuit[k]
        a2, b2 = circuit[k + 1]
        e1, r1, f1 = hid(a1, b1)
        e2, r2, f2 = hid(a2, b2)
        o1 = N - 0.5 if f1 else 0.5
        o2 = 0.5 if f2 else N - 0.5
        d = float(np.linalg.norm(ap(e1, r1, o1) - ap(e2, r2, o2)))
        ds.append(d)
        tot += max(3, math.ceil(d / 0.6))
    return tot, ds


best = []
random.seed(1)
circs = [euler_rand(arcs, 0) for _ in range(300)]
circs = [list(c) for c in {tuple(c) for c in circs}]
print("distinct circuits", len(circs))
for N in (80, 84, 88):
    res = []
    for signs in itertools.product([1, -1], repeat=6):
        for c in circs:
            tot, ds = evaluate(N, signs, c)
            res.append((tot, signs, N, max(ds)))
    res.sort()
    print(N, res[:3])
