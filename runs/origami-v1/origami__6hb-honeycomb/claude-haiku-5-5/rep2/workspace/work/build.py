import scadnano as sc

# ring of 6 honeycomb helices (cadnano coords); consecutive entries are neighbours, last-first too
pos = [(1,1),(0,1),(0,2),(1,2),(2,2),(2,1)]
MAXO = 126
helices = [sc.Helix(max_offset=MAXO, grid_position=p) for p in pos]

# scaffold: one strand, raster 5 -> 0 -> 1 -> 2 -> 3 -> 4 (ring minus edge 4-5 is NOT used; see below)
# scaffold directions: H0 fwd, H1 rev, H2 fwd, H3 rev, H4 fwd, H5 rev
scaf = sc.Strand(domains=[
    sc.Domain(5, False, 4, 125),
    sc.Domain(0, True, 4, 123),
    sc.Domain(1, False, 0, 123),
    sc.Domain(2, True, 0, 120),
    sc.Domain(3, False, 0, 120),
    sc.Loopout(length=8),
    sc.Domain(4, True, 7, 125),
], is_scaffold=True)

scaf_fwd = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False}
# scaffold coverage per helix (start,end)
cov = {0: (4,123), 1: (0,123), 2: (0,120), 3: (0,120), 4: (7,125), 5: (4,125)}

staples = []
for h in range(6):
    s, e = cov[h]
    staples.append(sc.Strand(domains=[sc.Domain(h, not scaf_fwd[h], s, e)]))

design = sc.Design(helices=helices, strands=[scaf] + staples, grid=sc.honeycomb)

# staple crossovers: (a, b, offsets) ; a's staple direction = not scaf_fwd[a]
xovers = {
    (0,1): [13,34,55,76,97],
    (1,2): [27,48,69,90,111],
    (2,3): [20,41,62,83,104],
    (3,4): [34,55,76,97],
    (4,5): [27,48,69,90,111],
    (5,0): [20,41,62,83,104],
}
for (a, b), offs in xovers.items():
    fa = not scaf_fwd[a]
    for o in offs:
        design.add_full_crossover(helix=a, helix2=b, offset=o, forward=fa)

def report():
    return [(s.dna_length(), s) for s in design.strands if not s.is_scaffold]

# break long staples with nicks until all <= 50 nt
for it in range(200):
    changed = False
    for s in list(design.strands):
        if s.is_scaffold or s.dna_length() <= 50:
            continue
        total = s.dna_length()
        target = total // 2
        # candidate domains: length >= 14, choose one whose midpoint is nearest to target
        best = None
        acc = 0
        for d in s.domains:
            if isinstance(d, sc.Loopout):
                acc += d.length
                continue
            L = d.end - d.start
            if L >= 14:
                mid = acc + L / 2
                if best is None or abs(mid - target) < best[0]:
                    best = (abs(mid - target), d, acc)
            acc += L
        if best is None:
            continue
        _, d, acc = best
        k = target - acc  # bases of this domain before the nick, in strand order
        k = max(6, min(d.end - d.start - 6, k))
        if d.forward:
            x = d.start + k
        else:
            x = d.end - k
        design.add_nick(helix=d.helix, offset=x, forward=d.forward)
        changed = True
        break
    if not changed:
        break

print("staple lengths:", sorted(s.dna_length() for s in design.strands if not s.is_scaffold))
print("scaffold length:", scaf.dna_length() if hasattr(scaf,'dna_length') else None)
for s in design.strands:
    if s.is_scaffold:
        print("scaffold strand length", s.dna_length())

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence and "?" in s.dna_sequence else None
design.write_scadnano_file(filename="designs/6hb.sc")
print("written")
