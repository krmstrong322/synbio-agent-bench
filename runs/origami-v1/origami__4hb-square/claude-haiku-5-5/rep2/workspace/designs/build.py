import scadnano as sc

# 2x2 square: helix 0 (0,0), 1 (1,0), 2 (1,1), 3 (0,1); ring 0-1-2-3-0
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]
SPAN = {0: (8, 70), 1: (8, 70), 2: (3, 70), 3: (3, 70)}
helices = [sc.Helix(max_offset=72, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square,
                   geometry=sc.Geometry(bases_per_turn=10.67))

# Scaffold: h0 reverse (top->bottom), h1 fwd, h2 rev, h3 fwd. Turns:
#   0->1 at bottom (offset 8), 1->2 at top (69), 2->3 at bottom (3).
scaf = {0: False, 1: True, 2: False, 3: True}
for h in range(4):
    lo, hi = SPAN[h]
    if scaf[h]:
        design.draw_strand(h, lo).move(hi - lo).as_scaffold()
        design.draw_strand(h, hi).move(lo - hi)   # staple, reverse
    else:
        design.draw_strand(h, hi).move(lo - hi).as_scaffold()
        design.draw_strand(h, lo).move(hi - lo)   # staple, forward

# scaffold turns (half crossovers at helix ends)
design.add_half_crossover(helix=0, helix2=1, offset=8, forward=False)
design.add_half_crossover(helix=1, helix2=2, offset=69, forward=True)
design.add_half_crossover(helix=2, helix2=3, offset=3, forward=False)

# staple crossovers (full): (helix a, helix b, offset, forward of staple on a)
XO = [
    (0, 1, 25, True), (0, 1, 57, True),
    (1, 2, 33, False),
    (2, 3, 41, True),
    (3, 0, 17, False), (3, 0, 49, False),
]
for a, b, o, f in XO:
    design.add_full_crossover(helix=a, helix2=b, offset=o, forward=f)

# extra nicks to keep staples <= ~50 nt
NICKS = [(1, 45, False), (3, 29, False), (0, 37, True)]
for h, o, f in NICKS:
    design.add_nick(helix=h, offset=o, forward=f)

design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/4hb_v1.sc")
for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
