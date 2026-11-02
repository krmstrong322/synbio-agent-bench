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

## Common failure modes

| Problem | Likely cause | Fix |
|---|---|---|
| strained crossover (a long backbone bond in 3D) | crossover at an offset where the backbones don't face | move it to an offset from `find_crossover_offsets` |
| staples likely to fall off in one region | short staple domains, or too few crossovers there | lengthen domains, move nicks, add crossovers |
| a helix barely attached | too few crossovers to its neighbours | add staple crossovers along that helix's whole length |
| twisted or bent structure | crossover spacing doesn't match the lattice's twist | check crossover spacing and twist (square lattice) |
| floppy structure | too few crossovers | add crossovers, especially near the ends |
| sheet splits along a seam | a line of nicks or a gap in crossovers between two rows | stagger nicks; connect every pair of rows along their whole overlap |

## Flat shapes (Rothemund-style 2D origami)

The simplest single-layer shape is a raster: parallel helices in **one row of the square lattice**, each
spanning the width of the shape at its height. That suits rectangles and disks. Shapes with sharp corners
hold together better when built from several domains (see below).

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

### Sharp corners: build the shape from domains

A single raster runs out of helix at a sharp corner: the rows near a triangle's apex or a star's tip are
only a few turns long, held by one or two crossovers, and they splay or fold over. Rothemund (2006) saw this
directly. His triangle made from three rectangular domains rarely came out equiangular. His **sharp
triangle**, made from three **trapezoids** with bridged seams, folded well (88% well formed).

The sharp triangle:
- **One trapezoid per edge.** Its helices run parallel to that outer edge and get shorter towards the
  centre. The slanted ends follow seams that bisect the triangle's corners (30° to each edge), so each row
  inward loses √3 × 2.6 nm ≈ 4.5 nm (≈13 bp) at each end. Rows pair up at alternating ends for the scaffold
  turns, so each slanted end is a 2-row staircase.
- **Leave the centre open.** Stop adding rows while the innermost row is still ≥ ~100 bp long. The outline
  is still a triangle, and no helix is short.
- **One scaffold through all three.** Raster each trapezoid and cross between trapezoids where two row ends
  meet: the outer rows at an outer corner, or the innermost rows at a corner of the hole. With an odd number
  of rows per trapezoid: trapezoid A from inner to outer row, cross at the outer corner into B, B from outer
  to inner, cross at the inner corner into C, C from inner to outer. Leave 2–3 unpaired scaffold bases at
  each crossing.
- **Bridge the seams.** Join the staple ending at one trapezoid's row end to the staple starting at the
  neighbouring trapezoid's matching row end, with a short poly-T loop (0–3 T, more towards the outer edge
  where the gap is wider). Without bridges the trapezoids hold only by stacking, and their tips fold inwards.

The same idea applies to other shapes with corners or narrow arms: split the shape into domains whose
helices run along the long direction of each part, and join the domains with the scaffold and bridging
staples (see the scadnano skill's helix groups).

## Wireframe polyhedra (DX edges)

Wireframe origami (DAEDALUS; Veneziano et al., 2016) builds a hollow polyhedron from edges that are each a
two-helix bundle (DX: two antiparallel helices side by side, joined by crossovers), meeting at vertices.

- **Edges.** Each edge is its own helix group with two rows (see the scadnano skill: orienting a group in 3D).
  Make each edge a whole number of turns at 10.5 bp/turn (42, 63, 84, 105 bp; 42 bp minimum) and use
  `bases_per_turn=10.5` for those groups. Lay the two helices side by side in the plane tangent to the
  surface: with t the edge direction and n the outward normal at the edge's midpoint (from the centre of the
  polyhedron), put row 1 on the side `n × t` of row 0.
- **Vertices.** Stop each helix short of the vertex point (about 4–5 nm where three edges meet at 60°) so the
  edges don't collide; the vertex point is where the edge lines would meet. Edges join at the vertex only through
  the scaffold and staples, with 2–3 unpaired bases (scaffold) or T (staples) bridging each hop.
- **Scaffold route.** Choose a spanning tree of the edges (for a tetrahedron, the three edges at one vertex).
  Keep right: on edge v→w the scaffold runs v→w on the helix on the right (seen from outside, looking along
  v→w) and w→v on the left one. Walking the route, at every vertex turn onto the next edge counter-clockwise
  (seen from outside). A tree edge is crossed twice, once along each helix. An edge outside the tree is
  entered from both ends: run to the middle, cross to the other helix (a scaffold crossover at a valid offset
  near mid-edge) and come back. The route then visits every helix once and closes into one loop; open it at a
  nick in the middle of an edge.
- **Edge staples.** Put staple crossovers between an edge's two helices every 21 bp (two turns), starting
  about 21 bp in from each end. Between two crossovers the staples close into a 42-nt ring; nick each ring
  once, mid-domain, giving 42-nt staples (about 10 + 21 + 11 nt).
- **Vertex staples.** At every scaffold hop between edges, join the staple on the edge the scaffold leaves to the
  staple on the edge it enters, with a short poly-T loop. Without these, the edges are held together only by the
  scaffold and the vertices are floppy. Give each arm at least ~10 bp: nick the U-shaped staple at an edge's end
  so that one arm of about 11 bp is left at the vertex, and let the vertex staple carry it across.
- **Check the geometry** before stapling: the helix ends that each hop joins should be 1–4 nm apart, and
  helices from different edges should be ≥ ~2.5 nm apart at the vertices. A hop much longer than that means a
  helix is on the wrong side (break the keep-right rule and the route crosses the vertex).

**Generate, don't hand-place.** Shapes with thousands of bases need loops: compute row extents, draw strands,
add crossovers at computed offsets, then nick. Print the row lengths and the total scaffold length from your
script so you can see that the shape and size come out as intended.
