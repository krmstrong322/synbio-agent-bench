import scadnano as sc

L = 72
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]  # 2x2 square: 0-1 horizontal, 1-2 vertical, 2-3 horizontal, 3-0 vertical
helices = [sc.Helix(max_offset=L, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square)  # default 10.5 bp/turn geometry

# scaffold domains: h0 fwd [1,67), h1 rev [1,67), h2 fwd [1,72), h3 rev [6,72)
SCAF = {0: (1, 67, True), 1: (1, 67, False), 2: (1, 72, True), 3: (6, 72, False)}
for h, (s, e, fwd) in SCAF.items():
    if fwd:
        design.draw_strand(h, s).move(e - s).as_scaffold()
    else:
        design.draw_strand(h, e).move(s - e).as_scaffold()

# scaffold junctions (half crossovers) at valid offsets
design.add_half_crossover(helix=0, helix2=1, offset=66, forward=True)   # A set
design.add_half_crossover(helix=1, helix2=2, offset=1, forward=False)   # B set
design.add_half_crossover(helix=2, helix2=3, offset=71, forward=True)   # R set

# staple tracks: opposite direction to scaffold, full length of each helix domain
for h, (s, e, fwd) in SCAF.items():
    if fwd:
        design.draw_strand(h, e).move(s - e)
    else:
        design.draw_strand(h, s).move(e - s)

# staple full crossovers: (helix a, helix b, offset, forward of strand on a)
XO = [
    (0, 1, 30, False), (0, 1, 51, False),
    (1, 2, 17, True), (1, 2, 38, True),
    (2, 3, 35, False), (2, 3, 56, False),
    (3, 0, 22, True), (3, 0, 43, True),
]
for a, b, o, f in XO:
    design.add_full_crossover(helix=a, helix2=b, offset=o, forward=f)

design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/v1.sc")
for s in design.strands:
    print("SCAF " if s.is_scaffold else "stap ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
