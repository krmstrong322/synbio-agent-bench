import scadnano as sc

N = 8
START, END = 0, 96
geom = sc.Geometry(bases_per_turn=10.67)
helices = [sc.Helix(max_offset=END, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=geom)

# scaffold: one strand, helix 0 fwd -> 1 rev -> 2 fwd ... 7 rev, turns at helix ends
scaf_doms = [sc.Domain(helix=h, forward=(h % 2 == 0), start=START, end=END) for h in range(N)]
design.add_strand(sc.Strand(domains=scaf_doms, is_scaffold=True))

# staples: full-length strand per helix, opposite to scaffold
for h in range(N):
    design.add_strand(sc.Strand(domains=[sc.Domain(helix=h, forward=(h % 2 == 1), start=START, end=END)]))

# staple crossovers, 16 bp spacing alternating sides along each helix
for h in range(N - 1):
    offsets = [9, 41, 73] if h % 2 == 0 else [25, 57]
    staple_fwd = (h % 2 == 1)
    for o in offsets:
        design.add_full_crossover(helix=h, helix2=h + 1, offset=o, forward=staple_fwd)

# nicks: split every staple longer than 48 nt into near-equal pieces
TARGET = 48
cuts_todo = []
for s in list(design.strands):
    if s.is_scaffold:
        continue
    L = s.dna_length()
    if L <= TARGET + 12:
        continue
    n_pieces = -(-L // TARGET)
    spacing = L / n_pieces
    targets = [round(spacing * k) for k in range(1, n_pieces)]
    # locate each target cumulative position inside a domain
    cum = 0
    doms = s.domains
    for t in targets:
        cum = 0
        for d in doms:
            dl = d.end - d.start
            if cum <= t < cum + dl:
                # position within domain
                within = t - cum
                # keep >=8 bp on both sides
                within = max(8, min(dl - 8, within))
                off = d.start + within
                cuts_todo.append((d.helix, off, d.forward))
                break
            cum += dl
        else:
            raise RuntimeError("no domain for cut")

for h, off, f in cuts_todo:
    design.add_nick(helix=h, offset=off, forward=f)

for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/build2.sc")
