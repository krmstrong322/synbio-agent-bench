import scadnano as sc
N = 8
L = 96
helices = [sc.Helix(max_offset=L, grid_position=(i, 0)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))

# scaffold raster: even helices forward (left->right), odd helices reverse
for h in range(N):
    if h % 2 == 0:
        design.draw_strand(h, 0).move(L).as_scaffold()
        design.draw_strand(h, L).move(-L)
    else:
        design.draw_strand(h, L).move(-L).as_scaffold()
        design.draw_strand(h, 0).move(L)

# scaffold turnarounds (half crossovers at helix ends)
for h in range(N - 1):
    if h % 2 == 0:
        design.add_half_crossover(helix=h, helix2=h + 1, offset=L - 1, forward=True)
    else:
        design.add_half_crossover(helix=h, helix2=h + 1, offset=0, forward=False)

# staple crossovers: pair (h,h+1) at offsets with 32-bp period, phase alternating
for h in range(N - 1):
    offs = [9, 41, 73] if h % 2 == 0 else [25, 57, 89]
    for o in offs:
        # strand on helix h running reverse (staple on even helices), forward on odd
        # choose direction so the crossover is on the staple of helix h
        fwd_h = (h % 2 == 1)  # staple direction on helix h: even->reverse, odd->forward
        design.add_full_crossover(helix=h, helix2=h + 1, offset=o, forward=fwd_h)

for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
design.write_scadnano_file(filename="designs/build.sc")
