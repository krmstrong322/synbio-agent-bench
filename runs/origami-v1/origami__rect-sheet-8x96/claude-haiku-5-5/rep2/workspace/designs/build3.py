import scadnano as sc

N = 8
START, END = 0, 96
geom = sc.Geometry(bases_per_turn=10.67)
helices = [sc.Helix(max_offset=END, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=geom)

scaf_doms = [sc.Domain(helix=h, forward=(h % 2 == 0), start=START, end=END) for h in range(N)]
design.add_strand(sc.Strand(domains=scaf_doms, is_scaffold=True))
for h in range(N):
    design.add_strand(sc.Strand(domains=[sc.Domain(helix=h, forward=(h % 2 == 1), start=START, end=END)]))

for h in range(N - 1):
    offsets = [9, 41, 73] if h % 2 == 0 else [25, 57]
    staple_fwd = (h % 2 == 1)
    for o in offsets:
        design.add_full_crossover(helix=h, helix2=h + 1, offset=o, forward=staple_fwd)

# iterative nicking: any staple longer than 48 nt gets a nick near its middle
# (in the interior of a domain, >=8 bp from each domain end)
MAXL = 48
for it in range(60):
    todo = None
    for s in design.strands:
        if s.is_scaffold or s.dna_length() <= MAXL:
            continue
        todo = s
        break
    if todo is None:
        break
    L = todo.dna_length()
    half = L // 2
    # candidate nicks: interior of any domain, leaving >=22 nt on both sides
    best = None
    cum = 0
    for d in todo.domains:
        dl = d.end - d.start
        for within in range(8, dl - 7):
            p = cum + within
            if 22 <= p <= L - 22:
                score = abs(p - half)
                if best is None or score < best[0]:
                    best = (score, d, d.start + within)
        cum += dl
    assert best is not None, "no valid nick"
    _, d, off = best
    design.add_nick(helix=d.helix, offset=off, forward=d.forward)

for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/build3.sc")
