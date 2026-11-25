import scadnano as sc
cells = [(2, -1), (2, 0), (2, 1), (2, 2), (3, -1), (3, 0), (3, 1), (3, 2), (4, -1), (4, 0), (4, 1), (4, 2)]
helices = [sc.Helix(max_offset=90, grid_position=c) for c in cells]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)
design.write_scadnano_file(filename="designs/draft.sc")
print("ok")
