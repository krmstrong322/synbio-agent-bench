import scadnano as sc

START, END = 0, 64
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]  # cycle 0-1-2-3-0 on the square lattice
helices = [sc.Helix(max_offset=END, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square,
                   geometry=sc.Geometry(bases_per_turn=10.67))

# Scaffold: h0 fwd -> h1 rev -> h2 fwd -> h3 rev (raster around the square); staples opposite.
for h in range(4):
    if h % 2 == 0:
        design.draw_strand(h, START).move(END - START).as_scaffold()
        design.draw_strand(h, END).move(START - END)
    else:
        design.draw_strand(h, END).move(START - END).as_scaffold()
        design.draw_strand(h, START).move(END - START)

# Scaffold turns (half crossovers at helix ends).
design.add_half_crossover(helix=0, helix2=1, offset=END - 1, forward=True)
design.add_half_crossover(helix=1, helix2=2, offset=START, forward=False)
design.add_half_crossover(helix=2, helix2=3, offset=END - 1, forward=True)

# Staple crossovers (full crossovers). Staples: h0 rev, h1 fwd, h2 rev, h3 fwd.
# (0,1): 9, 41   (1,2): 17, 49   (2,3): 25, 57   (3,0): 33
for o in (9, 41):
    design.add_full_crossover(helix=0, helix2=1, offset=o, forward=False)
for o in (17, 49):
    design.add_full_crossover(helix=1, helix2=2, offset=o, forward=True)
design.add_full_crossover(helix=2, helix2=3, offset=25, forward=False)
design.add_full_crossover(helix=3, helix2=0, offset=33, forward=True)

# Nicks to bring staples under 60 nt.
design.add_nick(helix=2, offset=37, forward=False)
design.add_nick(helix=0, offset=21, forward=False)
design.add_nick(helix=3, offset=45, forward=True)
design.add_nick(helix=1, offset=29, forward=True)

for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "fwd" if d.forward else "rev") for d in s.domains])

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence else None
design.write_scadnano_file(filename="designs/4hb_square.sc")
print("saved")
