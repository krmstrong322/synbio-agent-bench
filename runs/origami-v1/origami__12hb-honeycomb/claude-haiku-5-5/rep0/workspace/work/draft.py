import scadnano as sc
cells = [(h, v) for h in (3, 4, 5) for v in range(4)]
helices = [sc.Helix(max_offset=100, grid_position=c) for c in cells]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)
design.write_scadnano_file(filename="designs/draft.sc")
print(cells)
