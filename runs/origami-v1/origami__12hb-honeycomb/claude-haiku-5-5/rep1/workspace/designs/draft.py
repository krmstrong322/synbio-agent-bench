import scadnano as sc
pos = [(0,2),(0,1),(1,1),(2,1),(2,2),(1,2),(1,3),(0,3),(0,4),(1,4),(2,4),(2,3)]
helices = [sc.Helix(max_offset=84, grid_position=p) for p in pos]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)
design.write_scadnano_file(filename="designs/draft.sc")
print("ok")
