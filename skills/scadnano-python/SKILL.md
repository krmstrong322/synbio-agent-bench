---
name: scadnano-python
description: How to build DNA origami designs with the scadnano Python library - helices and lattices, drawing scaffold and staple strands, crossovers, nicks, M13 sequence assignment, insertions/deletions, saving .sc files. Read before writing any design script.
---

# scadnano Python API

`import scadnano as sc` (version 0.21). A worked example lives in
`/skills/scadnano-python/examples/three_helix_sheet.py`. Read it first.

## Coordinates

- A **helix** is indexed by an integer `idx` and spans offsets `[min_offset, max_offset)` (one offset = one base pair).
- A **domain** is a contiguous stretch of one strand on one helix: `sc.Domain(helix, forward, start, end)`. `end` is exclusive.
  `forward=True` means the strand runs 5'→3' left to right (increasing offset).
- Two domains on the same helix with opposite `forward` and overlapping offsets are base paired.
- A **strand** is a list of domains, 5'→3'. Consecutive domains on different helices form a **crossover**.

## Building a design

```python
helices = [sc.Helix(max_offset=200, grid_position=(h, v)) for (h, v) in positions]
design = sc.Design(helices=helices, strands=[], grid=sc.honeycomb)   # or sc.square
# square lattice convention: geometry=sc.Geometry(bases_per_turn=10.67)
```

- `grid_position=(h, v)`: `h` is the horizontal grid coordinate and `v` the vertical one.
  - **square**: neighbours of (h, v) are (h±1, v) and (h, v±1).
  - **honeycomb** (cadnano coordinates): every cell has 3 neighbours: (h±1, v) and one vertical neighbour.
    A compact 6-helix ring is `[(1,1),(0,1),(0,2),(1,2),(2,2),(2,1)]` (consecutive entries are neighbours, and the last is next to the first).
  - Don't guess adjacency. Call `find_crossover_offsets` on a draft: it reports `adjacent: true/false`.

### Drawing strands (StrandBuilder)

```python
design.draw_strand(helix, offset).move(n)        # domain [offset, offset+n), forward
design.draw_strand(helix, offset).move(-n)       # domain [offset-n, offset), reverse
design.draw_strand(0, 0).move(32).cross(1).move(-32)   # crossover to helix 1 at the same offset
design.draw_strand(0, 0).to(32).cross(1, offset=40).to(8)  # .to() goes to an absolute offset
design.draw_strand(0, 0).move(64).as_scaffold()  # mark as scaffold
```

`draw_strand` adds the strand to the design immediately. You can also build `sc.Strand([...domains...])`
objects and pass them as `strands=` to `sc.Design`. Mark the scaffold with `strand.set_scaffold()`.

### Editing: crossovers, nicks, ligation

A common workflow: draw full-length scaffold and staple domains on every helix, then cut and reconnect.

```python
design.add_nick(helix=h, offset=o, forward=f)          # cut between o-1 and o
design.add_full_crossover(helix=a, helix2=b, offset=o, forward=f)
#   two half crossovers between o-1 and o, on the strand on helix a with direction f
#   (the strand on helix b must run the opposite way). Auto-nicks if needed.
design.add_half_crossover(helix=a, helix2=b, offset=o, forward=f)
#   joins a strand END at offset o on helix a to a strand END at offset o on helix b.
#   Both strands must already end there (5'/3' ends adjacent).
design.ligate(helix=h, offset=o, forward=f)            # undo a nick
```

An `IllegalDesignError` from a crossover or nick call usually means no domain with that direction covers
that offset on one of the helices, or (for half crossovers) the strands don't end there. Check directions: scaffold and staple run opposite ways on each helix,
and neighbouring helices' scaffolds run opposite ways.

### Sequences

```python
design.assign_m13_to_scaffold()   # exactly one scaffold required; uses M13mp18 (p7249), rotation 5587
design.assign_m13_to_scaffold(rotation=0, variant=sc.M13Variant.p7249)
```

The scaffold takes the first N bases of M13 starting at `rotation`. Staples automatically get the complementary
sequence wherever they pair with the scaffold. Staple bases that pair with nothing get `?` and fail validation.
Assign sequences **last**, after all strand edits.

### Insertions / deletions (twist correction)

```python
design.add_deletion(helix=h, offset=o)          # skip one base pair at offset o (both strands)
design.add_insertion(helix=h, offset=o, length=1)
```

### Inspecting and saving

```python
for s in design.strands:
    print(s.is_scaffold, s.dna_length(), [(d.helix, d.start, d.end, d.forward) for d in s.domains])
design.write_scadnano_file(filename="designs/v1.sc")       # relative to the workspace root
design = sc.Design.from_scadnano_file("designs/v1.sc")
```

## Gotchas

- Domains of the same direction on the same helix must not overlap.
- `add_full_crossover` at offset `o` needs both strands to cover `o-1` and `o`.
- Put the scaffold's 5'/3' ends (and any seam) where a nick is acceptable. A scaffold that should be one
  strand must end up as exactly one `Strand` with `is_scaffold=True`.
- Print strand summaries after each step while developing. It's the fastest way to see what an edit did.
