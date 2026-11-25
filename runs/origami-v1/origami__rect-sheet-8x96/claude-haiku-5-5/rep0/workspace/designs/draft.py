import scadnano as sc
N = 8
L = 96
helices = [sc.Helix(max_offset=L, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))
for h in range(N):
    if h % 2 == 0:
        design.draw_strand(h, 0).move(L).as_scaffold()
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, L).move(-L).as_scaffold()
        design.draw_strand(h, 0).move(L)
design.write_scadnano_file(filename="designs/draft.sc")
