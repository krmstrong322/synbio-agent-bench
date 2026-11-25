import scadnano as sc
N = 8
START, END = 0, 96
helices = [sc.Helix(max_offset=END, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))
for h in range(N):
    if h % 2 == 0:
        design.draw_strand(h, START).move(END - START).as_scaffold()
        design.draw_strand(h, END).move(START - END)
    else:
        design.draw_strand(h, END).move(START - END).as_scaffold()
        design.draw_strand(h, START).move(END - START)
design.write_scadnano_file(filename="designs/draft.sc")
