import scadnano as sc

L = 72
pos = [(0, 0), (1, 0), (1, 1), (0, 1)]  # ring 0-1-2-3-0 in a 2x2 square
helices = [sc.Helix(max_offset=L, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))

for h in range(4):
    if h % 2 == 0:
        design.draw_strand(h, 0).move(L).as_scaffold()
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, L).move(-L).as_scaffold()
        design.draw_strand(h, 0).move(L)

design.write_scadnano_file(filename="designs/draft.sc")
