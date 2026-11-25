import scadnano as sc
N = 8
L = 98
XSET = {0: [9, 41, 73], 1: [25, 57, 89]}  # lower-helix crossover offsets by pair parity
X_ALL = {9, 25, 41, 57, 73, 89}

helices = [sc.Helix(max_offset=L, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))

for h in range(N):
    if h % 2 == 0:
        design.draw_strand(h, 0).move(L).as_scaffold()
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, L).move(-L).as_scaffold()
        design.draw_strand(h, 0).move(L)

for h in range(N - 1):
    if h % 2 == 0:
        design.add_half_crossover(helix=h, helix2=h + 1, offset=L - 1, forward=True)
    else:
        design.add_half_crossover(helix=h, helix2=h + 1, offset=0, forward=False)

for h in range(N - 1):
    fwd_h = (h % 2 == 1)
    for o in XSET[h % 2]:
        design.add_full_crossover(helix=h, helix2=h + 1, offset=o, forward=fwd_h)

def staple_len(s):
    return s.dna_length()

def split_options(s, helix, p):
    """lengths of the two pieces if strand s is nicked at (helix, p)."""
    before = 0
    for d in s.domains:
        if d.helix == helix and d.start < p < d.end:
            if d.forward:
                a = before + (p - d.start)
            else:
                a = before + (d.end - p)
            return a, s.dna_length() - a
        before += d.end - d.start
    return None

def find_nick(s):
    best = None
    for d in s.domains:
        for p in range(d.start + 1, d.end):
            if p in X_ALL or p == 0 or p >= L:
                continue
            if min(abs(p - x) for x in list(X_ALL) + [0, L]) < 7:
                continue
            r = split_options(s, d.helix, p)
            if r is None:
                continue
            a, b = r
            if a < 18 or b < 18:
                continue
            score = max(a, b)
            if max(a, b) > 60:
                score = 1000 + max(a, b)
            if best is None or score < best[0]:
                best = (score, d.helix, p, d.forward)
    return best

for it in range(100):
    long_ones = [s for s in design.strands if (not s.is_scaffold) and s.dna_length() > 60]
    if not long_ones:
        break
    s = max(long_ones, key=lambda x: x.dna_length())
    best = find_nick(s)
    if best is None:
        print("no nick for", s.dna_length())
        break
    _, h, p, f = best
    design.add_nick(helix=h, offset=p, forward=f)

design.assign_m13_to_scaffold()
lens = []
for s in design.strands:
    if not s.is_scaffold:
        lens.append(s.dna_length())
print("staple count", len(lens), "min", min(lens), "max", max(lens), "total", sum(lens))
for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
design.write_scadnano_file(filename="designs/origami_v2.sc")
