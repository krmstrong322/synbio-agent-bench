import scadnano as sc

L = 126
pos = [(1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]
helices = [sc.Helix(max_offset=L, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)
for h in range(6):
    if h % 2 == 0:
        design.draw_strand(h, 0).move(L).as_scaffold()
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, L).move(-L).as_scaffold()
        design.draw_strand(h, 0).move(L)
design.write_scadnano_file(filename="designs/draft.sc")
print("ok")
