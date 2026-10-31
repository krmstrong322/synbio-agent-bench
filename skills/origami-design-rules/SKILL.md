---
name: origami-design-rules
description: DNA origami design principles - honeycomb vs square lattice geometry, crossover spacing, scaffold routing, staple length and domain rules, seams, helix ends and twist. Read when planning a design or deciding how to fix one.
---

# DNA origami design rules

## Geometry

- B-DNA: ~10.5 bp per turn, 0.332 nm rise per bp, ~2 nm diameter. Packed helices sit ~2.25–2.6 nm apart.
- A crossover can only sit where the backbones of two neighbouring helices face each other. Use
  `find_crossover_offsets(design_path, helix, helix2)` on a draft design to get the valid offsets for
  each strand direction. Don't hand-derive them.

### Honeycomb lattice (3D bundles: 6HB, 12HB, 24HB, ...)
- Each helix has up to 3 neighbours, 120° apart. At 10.5 bp/turn, 7 bp = 240°, so crossovers to the 3
  neighbours fall 7 bp apart and each pair repeats every **21 bp** (2 turns).
- Use scadnano's default geometry (10.5 bp/turn). No twist correction needed.

### Square lattice (flat sheets, square-section bundles)
- Each helix has up to 4 neighbours, 90° apart. By convention, designs assume **10.67 bp/turn**
  (`geometry=sc.Geometry(bases_per_turn=10.67)`), so crossovers to one neighbour repeat every **32 bp**
  (3 turns). In a flat sheet, crossovers to the upper and lower neighbours alternate every **16 bp**.
- Real DNA is 10.5 bp/turn, so a 10.67 design is under-twisted and the structure develops a global twist.
  Optional correction: one deletion every ~48 bp along each helix (`design.add_deletion`).

## Scaffold routing

- The scaffold is one continuous strand (a contiguous M13 fragment) that visits every helix.
- Neighbouring helices carry the scaffold in **opposite directions** (antiparallel).
- A raster path is standard: run along helix 0, jump to helix 1 at the end, come back, jump to helix 2, and
  so on. Scaffold crossovers at the helix ends are half crossovers. Pick end offsets where
  `find_crossover_offsets` allows a half crossover, which means helix ends may differ by a few bp.
- Optional internal scaffold crossovers add rigidity. Place them only at valid offsets.
- The scaffold's free 5' and 3' ends should be at a helix end.

## Staples

- Length **18–60 nt** (typical 32–49). Very short staples (<16 nt) melt; very long ones (>60) synthesize poorly.
- Each staple should span **2–3 helices** (1–2 crossovers). Crossovers between helices are what hold the
  shape together. Every pair of adjacent helices needs staple crossovers along its whole length, roughly
  every 21 bp (honeycomb) or 32 bp (square, alternating neighbours every 16 bp).
- Each staple domain (segment on one helix) should be **≥7 bp**, and every staple should have at least one
  domain ≥14 bp to anchor it. Nick placement controls domain lengths: put the nick between two crossovers,
  not next to one.
- Stagger nicks between neighbouring helices (a "brick" pattern) so no continuous seam of nicks runs through
  the structure.
- At helix ends, merge short leftover staple pieces into a neighbouring staple instead of leaving <16 nt fragments.
- Every scaffold base should be paired with a staple (coverage ≥95%), except deliberate single-stranded loops.

## Common failure modes (and what simulation shows)

| Symptom in `simulate_design` | Likely cause | Fix |
|---|---|---|
| `initial_backbone.worst_bonds` > 2.6 nm at a crossover | crossover at an offset where backbones don't face | move it to an offset from `find_crossover_offsets` |
| low base-pair occupancy in one region | short staple domains, or too few crossovers there | lengthen domains, move nicks, add crossovers |
| low occupancy along a whole helix | helix barely connected to its neighbours | add staple crossovers to that helix |
| large RMSD to design with low RMSF | the structure is stable but strained (twisted or bent) | check crossover spacing and twist (square lattice) |
| large RMSF | the structure is floppy: too few crossovers | add crossovers, especially near the ends |

## Flat shapes (Rothemund-style 2D origami)

Single-layer shapes (rectangles, triangles, disks, stars) are rasters: parallel helices in **one row of
the square lattice**, each spanning the width of the shape at its height.

**Sizing.** Use 0.34 nm per bp along a helix and 2.6 nm between neighbouring helices. A shape of area A
needs about A / (0.34 × 2.6) bp of scaffold, so the scaffold length sets the area. Choose the number of rows
from the shape's height (height / 2.6 nm). Give row k a length of ≈ width(y_k) / 0.34 bp, where y_k is that
row's height in the shape. Then adjust row lengths so the total matches the scaffold length.

**Scaffold path.** Use a raster: row 0 left→right, jump to row 1, row 1 right→left, and so on. Each jump is a
half crossover at a row end. Consecutive rows therefore share an end on alternating sides, so the shape's
edge becomes a staircase. Row ends must be at offsets where `find_crossover_offsets` allows a half crossover
for that strand direction. Snap each row's ends to the nearest allowed offset.

**Staples.** Use the square-lattice convention (`bases_per_turn=10.67`). Staple crossovers alternate between
the upper and lower neighbour every 16 bp. A typical interior staple is 32 nt: 8 bp + 16 bp + 8 bp over three
helices, or 16 + 16 over two. Where a neighbouring row is shorter, staples at the edge can only cross one way.
Merge or extend those staples so none falls below 16 nt.

**Weak points.** Narrow features (star tips, triangle corners) are held by only a few crossovers. Make sure
every row is connected to both neighbours wherever they overlap, along its whole length.

**Generate, don't hand-place.** Shapes with thousands of bases need loops: compute row extents, draw strands,
add crossovers at computed offsets, then nick. Check `shape_fit` (IoU) from `validate_design_file` after every change.
