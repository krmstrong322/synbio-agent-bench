import scadnano as sc

L = 126
pos = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]
helices = [sc.Helix(max_offset=L, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)

# scaffold raster: 0 fwd, 1 rev, 2 fwd, 3 rev, 4 fwd, 5 rev
d = design.draw_strand(0, 0).move(L).cross(1).move(-L).cross(2).move(L).cross(3).move(-L).cross(4).move(L).cross(5).move(-L)
d.as_scaffold()

# full-length staples, opposite to scaffold on each helix
for h in range(6):
    if h % 2 == 0:
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, 0).move(L)

design.write_scadnano_file(filename="designs/draft.sc")
print("ok", len(design.strands))
