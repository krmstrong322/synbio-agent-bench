import sys, json, random, itertools
sys.path.insert(0, '/scripts')
from geom import *

rng = random.Random(7)
best = None
perm_list = list(itertools.permutations(range(3)))
for trial in range(60000):
    s_idx = [rng.randrange(4) for _ in range(12)]
    bb = [rng.randrange(2) for _ in range(12)]
    perms = [rng.choice(perm_list) for _ in range(8)]
    single, cost, nxt, pts = evaluate(s_idx, bb, perms)
    if not single:
        continue
    if best is None or cost < best[0]:
        best = (cost, s_idx, bb, perms, nxt)
# local improvement: single-site moves keeping single cycle
cur = best
improved = True
it = 0
while improved and it < 30:
    improved = False; it += 1
    cost0, s0, b0, p0, _ = cur
    for e in range(12):
        for val in range(4):
            s1 = list(s0); s1[e] = val
            single, c, nx, _ = evaluate(s1, b0, p0)
            if single and c < cost0 - 1e-9:
                cur = (c, s1, b0, p0, nx); cost0, s0 = c, s1; improved = True
        for val in range(2):
            b1 = list(b0); b1[e] = val
            single, c, nx, _ = evaluate(s0, b1, p0)
            if single and c < cost0 - 1e-9:
                cur = (c, s0, b1, p0, nx); cost0, b0 = c, b1; improved = True
    for v in range(8):
        for pm in perm_list:
            p1 = list(p0); p1[v] = pm
            single, c, nx, _ = evaluate(s0, b0, p1)
            if single and c < cost0 - 1e-9:
                cur = (c, s0, b0, p1, nx); cost0, p0 = c, p1; improved = True
cost, s_idx, bb, perms, nxt = cur
single, cost, nxt, pts = evaluate(s_idx, bb, perms)
print('cost', cost, 'single', single)
# junction lengths
tot = []
for h in range(24):
    v, p = pts[(h, 'arr')]
    h2 = nxt[h]
    v2, p2 = pts[(h2, 'dep')]
    tot.append(float(np.linalg.norm(p - p2)))
print('junction distances', [round(x, 2) for x in tot])
json.dump({'s_idx': s_idx, 'bb': bb, 'perms': perms, 'nxt': nxt}, open('scripts/config.json', 'w'))
