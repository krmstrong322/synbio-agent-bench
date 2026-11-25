import scadnano as sc

START, END = 0, 64
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]  # cycle 0-1-2-3-0, all neighbours
helices = [sc.Helix(max_offset=END, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))

for h in range(4):
    if h % 2 == 0:
        design.draw_strand(h, START).move(END - START).as_scaffold()
        design.draw_strand(h, END).move(START - END)
    else:
        design.draw_strand(h, END).move(START - END).as_scaffold()
        design.draw_strand(h, START).move(END - START)

design.add_half_crossover(helix=0, helix2=1, offset=END-1, forward=True)
design.add_half_crossover(helix=1, helix2=2, offset=START, forward=False)
design.add_half_crossover(helix=2, helix2=3, offset=END-1, forward=True)

design.write_scadnano_file(filename="designs/draft4hb.sc")
for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "fwd" if d.forward else "rev") for d in s.domains])
