import scadnano as sc

def try_top(T):
    # helix0 fwd domain [0,T), helix1 rev domain [0,T): top ends at T-1
    helices = [sc.Helix(max_offset=96, grid_position=(i, 0)) for i in range(2)]
    d = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))
    d.draw_strand(0, 0).move(T).as_scaffold()
    d.draw_strand(1, T).move(-T)
    d.draw_strand(0, T).move(96 - T)
    d.draw_strand(1, 0).move(T)
    try:
        d.add_half_crossover(helix=0, helix2=1, offset=T - 1, forward=True)
        return "ok"
    except Exception as e:
        return "ERR " + str(e)[:120]

for T in range(86, 97):
    print("top T", T, try_top(T))
