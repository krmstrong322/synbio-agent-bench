import scadnano as sc
L = 96
def build():
    helices = [sc.Helix(max_offset=L, grid_position=(i, 0)) for i in range(2)]
    design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))
    design.draw_strand(0, 0).move(L).as_scaffold()
    design.draw_strand(0, L).move(-L)
    design.draw_strand(1, L).move(-L).as_scaffold()
    design.draw_strand(1, 0).move(L)
    return design
for o in [95, 96, 94, 89, 88, 93, 0, 1, 2, 3]:
    d = build()
    try:
        d.add_half_crossover(helix=0, helix2=1, offset=o, forward=True)
        print(o, "ok", [(s.is_scaffold, [(x.helix,x.start,x.end,x.forward) for x in s.domains]) for s in d.strands][:2])
    except Exception as e:
        print(o, "err", str(e)[:150])
