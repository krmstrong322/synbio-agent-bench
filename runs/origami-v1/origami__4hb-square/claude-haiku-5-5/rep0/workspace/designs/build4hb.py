import scadnano as sc

# Square 2x2 bundle: A(0,0)  B(1,0)  C(1,1)  D(0,1)  (ring A-B-C-D-A)
# Offsets are shifted by +32 from the lattice-validated set (32 bp = one full period).
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]
helices = [sc.Helix(max_offset=96, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square,
                   geometry=sc.Geometry(bases_per_turn=10.67))

# Scaffold directions: A fwd, B rev, C fwd, D rev. Staples opposite.
span = {0: (26, 90), 1: (22, 90), 2: (22, 84), 3: (17, 84)}
for h in range(4):
    lo, hi = span[h]
    if h in (0, 2):   # scaffold forward, staple reverse
        design.draw_strand(h, lo).move(hi - lo).as_scaffold()
        design.draw_strand(h, hi).move(lo - hi)
    else:             # scaffold reverse, staple forward
        design.draw_strand(h, hi).move(lo - hi).as_scaffold()
        design.draw_strand(h, lo).move(hi - lo)

# Scaffold turns (half crossovers at helix ends)
design.add_half_crossover(helix=0, helix2=1, offset=89, forward=True)   # A top -> B top
design.add_half_crossover(helix=1, helix2=2, offset=22, forward=False)  # B bottom -> C bottom
design.add_half_crossover(helix=2, helix2=3, offset=83, forward=True)   # C top -> D top

# Staple crossovers (offset = base index o, crossing between o-1 and o)
design.add_full_crossover(helix=0, helix2=1, offset=41, forward=False)
design.add_full_crossover(helix=0, helix2=1, offset=73, forward=False)
design.add_full_crossover(helix=1, helix2=2, offset=49, forward=True)
design.add_full_crossover(helix=2, helix2=3, offset=57, forward=False)
design.add_full_crossover(helix=3, helix2=0, offset=65, forward=True)

# Nicks to bring staples into 18-60 nt
design.add_nick(helix=0, offset=53, forward=False)
design.add_nick(helix=1, offset=57, forward=True)

for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])

design.assign_m13_to_scaffold()
design.write_scadnano_file(filename="designs/build_v2.sc")
