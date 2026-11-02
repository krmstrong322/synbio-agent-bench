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

### Helix groups (domains whose helices point in different directions)

A flat sheet can be built from several domains, each a `HelixGroup` with its own lattice, position and
orientation. Helices run along +z; on the square grid, `grid_position=(0, j)` puts row j at y = j × spacing.
`pitch` (degrees) rotates a group within that y–z plane: pitch 120 turns the helix direction from +z to
(y = 0.866, z = −0.5). Keep x = 0 and yaw/roll at 0 so every group stays in the same plane.

```python
geom = sc.Geometry(bases_per_turn=10.67, rise_per_base_pair=0.34, helix_radius=1.0, inter_helix_gap=0.6)  # 2.6 nm apart
groups = {
    "A": sc.HelixGroup(grid=sc.square, pitch=0,   position=sc.Position3D(x=0, y=1.3,  z=0.0),  geometry=geom),
    "B": sc.HelixGroup(grid=sc.square, pitch=120, position=sc.Position3D(x=0, y=-0.65, z=94.9), geometry=geom),
}
helices = [sc.Helix(grid_position=(0, j), group=g, max_offset=300) for g in groups for j in range(7)]
design = sc.Design(helices=helices, groups=groups, strands=[])   # helix indices follow the list order
```

A group's `position` is where its offset 0 on row 0 sits, in nm. Strands may run through several groups.
Build a strand from explicit domains, where consecutive domains on different helices form a crossover:

```python
design.add_strand(sc.Strand(domains=[
    sc.Domain(helix=6, forward=True, start=100, end=290),
    sc.Loopout(length=3),                                  # unpaired bases bridging a gap between domains
    sc.Domain(helix=7, forward=True, start=12, end=280),
], is_scaffold=True))
```

`find_crossover_offsets` compares the same offset on two helices, so use it between rows of one group. Where
two groups meet, join row ends directly. A `Loopout` in a staple gets `?` bases from `assign_m13_to_scaffold`;
set them to T afterwards: `s.set_dna_sequence(s.dna_sequence.replace("?", "T"))`.

### Orienting a group in 3D (yaw, pitch, roll)

For 3D shapes (wireframe edges, bundles at angles) give each group `yaw`, `pitch` and `roll` in degrees.
scadnano applies yaw, then pitch, then roll, and the helices end up running along

    forward = (cos(pitch)·sin(yaw), sin(pitch), cos(pitch)·cos(yaw))

with roll turning the grid rows about that direction. To place a group from the directions you want:

```python
import math
import numpy as np

def group_angles(forward, side):
    """(yaw, pitch, roll) in degrees: helices run along unit vector `forward`; grid row 1 sits on the `side`
    of row 0 (a unit vector perpendicular to forward)."""
    f, s = np.asarray(forward, float), np.asarray(side, float)
    yaw, pitch = math.atan2(f[0], f[2]), math.asin(max(-1.0, min(1.0, f[1])))
    x0 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    y0 = np.array([-math.sin(pitch) * math.sin(yaw), math.cos(pitch), -math.sin(pitch) * math.cos(yaw)])
    roll = math.atan2(-s.dot(x0), s.dot(y0))
    return tuple(math.degrees(a) for a in (yaw, pitch, roll))

yaw, pitch, roll = group_angles(forward=t, side=side)
group = sc.HelixGroup(grid=sc.square, position=sc.Position3D(*p0), yaw=yaw, pitch=pitch, roll=roll,
                      geometry=sc.Geometry(bases_per_turn=10.5, rise_per_base_pair=0.34,
                                           helix_radius=1.0, inter_helix_gap=0.6))
```

With that group, the axis of row j at offset o is at `p0 + j*2.6*side + o*0.34*forward` (nm). Print these
points for every helix end and check that ends meant to meet are 1–4 nm apart and that no two groups' helices
come closer than ~2.5 nm. Strands may jump between groups anywhere: consecutive domains on different helices
make the jump, and a `Loopout` between them adds unpaired bases to bridge the gap.

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
