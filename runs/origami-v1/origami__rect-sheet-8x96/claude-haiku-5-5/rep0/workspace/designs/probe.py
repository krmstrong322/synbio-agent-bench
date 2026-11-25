import scadnano as sc
L = 96
def build(offs_fwd0, offs_rev0):
    helices = [sc.Helix(max_offset=L, grid_position=(i, 0)) for i in range(2)]
    design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))
    design.draw_strand(0, 0).move(L).as_scaffold()
    design.draw_strand(0, L).move(-L)
    design.draw_strand(1, L).move(-L).as_scaffold()
    design.draw_strand(1, 0).move(L)
    return design
for o in [14, 25, 9, 30]:
    d = build([], [])
    try:
        d.add_full_crossover(helix=0, helix2=1, offset=o, forward=True)
        print(o, "fwd ok")
    except Exception as e:
        print(o, "fwd err", str(e)[:150])
    d = build([], [])
    try:
        d.add_full_crossover(helix=0, helix2=1, offset=o, forward=False)
        print(o, "rev ok")
    except Exception as e:
        print(o, "rev err", str(e)[:150])
