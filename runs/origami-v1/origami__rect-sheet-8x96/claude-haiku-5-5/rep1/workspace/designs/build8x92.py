import scadnano as sc

N = 8
LO, HI = 8, 100          # every helix spans [8, 100) = 92 bp
helices = [sc.Helix(max_offset=110, grid_position=(0, i)) for i in range(N)]
design = sc.Design(helices=helices, strands=[], grid=sc.square,
                   geometry=sc.Geometry(bases_per_turn=10.67))

# scaffold: one serpentine strand through all helices
scaf_domains = []
for h in range(N):
    scaf_domains.append(sc.Domain(helix=h, forward=(h % 2 == 0), start=LO, end=HI))
design.add_strand(sc.Strand(domains=scaf_domains, is_scaffold=True))

# staples: one full-length strand per helix, opposite to the scaffold
for h in range(N):
    if h % 2 == 0:
        design.draw_strand(h, HI).move(LO - HI)   # reverse
    else:
        design.draw_strand(h, LO).move(HI - LO)   # forward

# staple crossovers between neighbours. Pair (h, h+1):
#   h even -> sites 30, 62 ; h odd -> sites 46, 78  (valid full-crossover offsets for this lattice)
for h in range(N - 1):
    sites = (30, 62) if h % 2 == 0 else (46, 78)
    for s in sites:
        fwd = (h % 2 == 1)   # staple on helix h: fwd for odd h, rev for even h
        design.add_full_crossover(helix=h, helix2=h + 1, offset=s, forward=fwd)

# nicks to split the long staple chains into pieces of 32-56 nt
#   chain A (172 nt): helix 1 @70 (F), helix 3 @70 (F), helix 6 @70 (R)
#   chain B (256 nt): helix 3 @54 (F), helix 0 @46 (R), helix 3 @38 (F), helix 6 @38 (R)
design.add_nick(helix=1, offset=70, forward=True)
design.add_nick(helix=3, offset=70, forward=True)
design.add_nick(helix=6, offset=70, forward=False)
design.add_nick(helix=3, offset=54, forward=True)
design.add_nick(helix=0, offset=46, forward=False)
design.add_nick(helix=3, offset=38, forward=True)
design.add_nick(helix=6, offset=38, forward=False)
design.add_nick(helix=6, offset=54, forward=False)   # closes the 256-nt ring into 48-nt pieces

design.assign_m13_to_scaffold()
for s in design.strands:
    if not s.is_scaffold:
        s.set_dna_sequence(s.dna_sequence.replace("?", "T")) if s.dna_sequence and "?" in s.dna_sequence else None

design.write_scadnano_file(filename="designs/rect_sheet_8x92.sc")
for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "F" if d.forward else "R") for d in s.domains])
