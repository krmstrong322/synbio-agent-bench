import scadnano as sc

# 6-helix ring on the honeycomb lattice
POS = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]
N = 6
# scaffold span [start, end) and direction per helix (even = forward)
S = [29, 22, 22, 8, 8, 1]
E = [155, 155, 141, 141, 127, 127]
FWD = [True, False, True, False, True, False]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
# staple crossover offsets per ring pair (found by anneal search; valid residues mod 21)
X = {
    (0, 1): [55, 76, 108, 129],
    (1, 2): [38, 80, 122],
    (2, 3): [41, 62, 94, 115],
    (3, 4): [24, 45, 76, 108],
    (4, 5): [17, 48, 69, 101],
    (5, 0): [52, 83, 104],
}

design = sc.Design(helices=[sc.Helix(max_offset=160, grid_position=POS[i]) for i in range(N)],
                   strands=[], grid=sc.honeycomb)

# scaffold: raster 0 -> 1 -> ... -> 5, junctions at the helix ends
dom = [sc.Domain(helix=h, forward=FWD[h], start=S[h], end=E[h]) for h in range(N)]
design.add_strand(sc.Strand(domains=dom, is_scaffold=True))

# boundaries per helix
B = [dict() for _ in range(N)]
for pair in PAIRS:
    for p in X[pair]:
        B[pair[0]][p] = pair
        B[pair[1]][p] = pair


def prev_boundary(h, p):
    return max([q for q in B[h] if q < p] + [S[h]])


staple_count = 0
lengths = []
# crossover staples: piece on a and piece on b, both ending at p
for pair in PAIRS:
    a, b = pair
    for p in X[pair]:
        pa = prev_boundary(a, p)
        pb = prev_boundary(b, p)
        fa = not FWD[a]           # staple direction on a (opposite to scaffold)
        da = sc.Domain(helix=a, forward=fa, start=pa, end=p)
        db = sc.Domain(helix=b, forward=not fa, start=pb, end=p)
        doms = [da, db] if fa else [db, da]   # the forward-going piece comes first
        st = sc.Strand(domains=doms)
        design.add_strand(st)
        lengths.append(st.dna_length())
        staple_count += 1

# standalone end pieces: from last boundary to the helix end
for h in range(N):
    last = max(list(B[h]) + [S[h]])
    fh = not FWD[h]
    st = sc.Strand(domains=[sc.Domain(helix=h, forward=fh, start=last, end=E[h])])
    design.add_strand(st)
    lengths.append(st.dna_length())
    staple_count += 1

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold and s.dna_sequence and "?" in s.dna_sequence:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T"))
design.write_scadnano_file(filename="designs/6hb.sc")

scaf = [s for s in design.strands if s.is_scaffold][0]
print("scaffold length", scaf.dna_length())
print("staples", staple_count, "min", min(lengths), "max", max(lengths))
print("crossovers per pair", {k: len(v) for k, v in X.items()})
