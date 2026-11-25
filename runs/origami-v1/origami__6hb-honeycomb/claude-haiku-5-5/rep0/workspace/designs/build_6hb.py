import scadnano as sc

L = 126
pos = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]   # closed honeycomb ring
helices = [sc.Helix(max_offset=L, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)

# ---- scaffold: raster 0 fwd, 1 rev, 2 fwd, 3 rev, 4 fwd, 5 rev (756 nt)
sc_strand = (design.draw_strand(0, 0).move(L).cross(1).move(-L).cross(2).move(L)
             .cross(3).move(-L).cross(4).move(L).cross(5).move(-L))
sc_strand.as_scaffold()

# ---- full-length staples: opposite to scaffold on each helix
# even helices: staple reverse; odd helices: staple forward
for h in range(6):
    if h % 2 == 0:
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, 0).move(L)

# ---- staple crossovers, one family per adjacent pair (offsets from find_crossover_offsets)
# (a, b, forward-direction of staple on a, offsets)
pairs = [
    (0, 1, False, [13, 34, 55, 76, 97]),
    (1, 2, True,  [27, 48, 69, 90, 111]),
    (2, 3, False, [20, 41, 62, 83, 104]),
    (3, 4, True,  [13, 34, 55, 76, 97]),
    (4, 5, False, [27, 48, 69, 90, 111]),
    (5, 0, True,  [20, 41, 62, 83, 104]),
]
for a, b, fwd, offs in pairs:
    for o in offs:
        design.add_full_crossover(helix=a, helix2=b, offset=o, forward=fwd)

def show(tag):
    print(tag)
    for s in design.strands:
        if not s.is_scaffold:
            print(" len", s.dna_length(), [(d.helix, d.start, d.end, d.forward) for d in s.domains])

# ---- nick long staples so each is <= 48 nt
MAXLEN = 48
for _ in range(200):
    changed = False
    for s in list(design.strands):
        if s.is_scaffold or s.dna_length() <= MAXLEN:
            continue
        cands = [d for d in s.domains if d.end - d.start >= 14 and d.start > 0 and d.end - d.start >= 14]
        if not cands:
            continue
        # choose domain whose midpoint is closest to the middle of the strand
        d = min(cands, key=lambda d: abs((d.start + d.end) / 2 - L / 2))
        mid = d.start + (d.end - d.start) // 2
        design.add_nick(helix=d.helix, offset=mid, forward=d.forward)
        changed = True
        break
    if not changed:
        break

lens = [s.dna_length() for s in design.strands if not s.is_scaffold]
print("n staples", len(lens), "min", min(lens), "max", max(lens))
scaf = [s for s in design.strands if s.is_scaffold][0]
print("scaffold len", scaf.dna_length())

design.assign_m13_to_scaffold(rotation=5587)
for s in design.strands:
    if not s.is_scaffold and s.dna_sequence:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
design.write_scadnano_file(filename="designs/6hb_v1.sc")
print("saved")
