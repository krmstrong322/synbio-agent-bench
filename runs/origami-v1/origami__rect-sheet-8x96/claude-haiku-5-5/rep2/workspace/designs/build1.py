import scadnano as sc

N = 8
START, END = 0, 96
geom = sc.Geometry(bases_per_turn=10.67)
helices = [sc.Helix(max_offset=END, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=geom)

# scaffold: one strand, helix 0 fwd -> 1 rev -> 2 fwd ... 7 rev, turns at helix ends
scaf_doms = []
for h in range(N):
    fwd = (h % 2 == 0)
    scaf_doms.append(sc.Domain(helix=h, forward=fwd, start=START, end=END))
design.add_strand(sc.Strand(domains=scaf_doms, is_scaffold=True))

# staples: full-length strand per helix, opposite to scaffold
for h in range(N):
    fwd = (h % 2 == 1)
    design.add_strand(sc.Strand(domains=[sc.Domain(helix=h, forward=fwd, start=START, end=END)]))

# staple crossovers: pair (h,h+1) uses B={9,41,73} if h even, A={25,57,89} if h odd
for h in range(N - 1):
    offsets = [9, 41, 73] if h % 2 == 0 else [25, 57, 89]
    staple_fwd = (h % 2 == 1)
    for o in offsets:
        design.add_full_crossover(helix=h, helix2=h + 1, offset=o, forward=staple_fwd)

for s in design.strands:
    if not s.is_scaffold:
        print("staple", s.dna_length(), [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
print("scaffold", design.strands[0].dna_length() if design.strands[0].is_scaffold else None)
design.write_scadnano_file(filename="designs/build1.sc")
