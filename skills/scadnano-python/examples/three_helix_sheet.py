"""Worked example: a 3-helix flat sheet on the square lattice (59 bp per helix).

Shows the full pattern: helices -> scaffold + staple strands -> scaffold
crossovers that join the scaffold into one strand -> staple crossovers ->
nicks to set staple lengths -> M13 sequence -> save.

Crossover offsets were taken from find_crossover_offsets on this helix layout.
"""

import scadnano as sc

START, END = 1, 60  # every helix spans offsets [1, 60)

# 1. Helices. Square lattice; grid_position=(h, v). Default positions put helix i at (0, i).
#    Square-lattice designs conventionally assume 10.67 bp/turn so that crossovers repeat
#    every 32 bp (3 turns); see the origami-design-rules skill.
helices = [sc.Helix(max_offset=72, grid_position=(0, i)) for i in range(3)]
design = sc.Design(helices=helices, strands=[], grid=sc.square, geometry=sc.Geometry(bases_per_turn=10.67))

# 2. Scaffold runs forward on even helices and reverse on odd helices (neighbours must be
#    antiparallel). Staples run the other way. draw_strand(helix, offset).move(n) makes a
#    domain from offset to offset+n (negative n = reverse strand).
for h in range(3):
    scaf_forward = h % 2 == 0
    if scaf_forward:
        design.draw_strand(h, START).move(END - START).as_scaffold()
        design.draw_strand(h, END).move(START - END)  # staple, reverse
    else:
        design.draw_strand(h, END).move(START - END).as_scaffold()
        design.draw_strand(h, START).move(END - START)  # staple, forward

# 3. Join the scaffold into one strand with half crossovers at helix ends.
#    Helix 0 (forward) ends at offset 59 -> jump to helix 1 there.
#    Helix 1 (reverse) ends at offset 1 -> jump to helix 2 there.
#    find_crossover_offsets lists 59 and 1 as half-crossover offsets for these strands.
design.add_half_crossover(helix=0, helix2=1, offset=59, forward=True)
design.add_half_crossover(helix=1, helix2=2, offset=1, forward=False)

# 4. Staple crossovers (full crossovers = two half crossovers + a nick) at valid offsets.
#    Helix 0 staples are reverse; helix 1 staples are forward. On a sheet, crossovers to the
#    upper and lower neighbour alternate every 16 bp (1.5 turns): 22 (0-1), 38 (1-2), 54 (0-1).
for offset in (22, 54):
    design.add_full_crossover(helix=0, helix2=1, offset=offset, forward=False)
design.add_full_crossover(helix=1, helix2=2, offset=38, forward=True)

# 5. Nicks break long staples. add_nick(helix, offset, forward) cuts between offset-1 and offset.
design.add_nick(helix=0, offset=38, forward=False)
design.add_nick(helix=2, offset=22, forward=False)

# (The 12-nt edge staple is too short for a real design; see origami-design-rules.)

# 6. Sequences: the scaffold gets M13mp18 (first N bases from the rotation); staples get the complement.
design.assign_m13_to_scaffold()

# 7. Save (relative path: the working directory is the workspace root).
design.write_scadnano_file(filename="designs/three_helix_sheet.sc")
for s in design.strands:
    print("scaffold" if s.is_scaffold else "staple  ", s.dna_length(),
          [(d.helix, d.start, d.end, "fwd" if d.forward else "rev") for d in s.domains])
