---
name: oxdna-testing
description: How to test a DNA origami design with the simulate_design tool (oxDNA2 coarse-grained MD) and interpret base-pair occupancy, RMSF, RMSD, extents, shape fidelity and backbone-bond reports. Read before your first simulation.
---

# Testing designs with oxDNA

`simulate_design(design_path)` exports the design to oxDNA (oxDNA2 model, 1 M NaCl, 25 °C) and runs:

1. **MC relaxation** with capped backbone forces, which pulls the idealized scadnano geometry together.
2. **MD relaxation** (capped forces, damped thermostat).
3. **Production MD**. Metrics are computed over the second half of the trajectory.

The simulation budget per task is small, and each run takes minutes. Always clear every error from
`validate_design_file` first.

## Reading the output

- `ok: false` + `failed_stage`: the simulation failed. `export` means scadnano couldn't convert the design.
  A crash in `relax_*` usually means very badly placed crossovers or strands that overlap in space. See `log_tail`.
- `initial_backbone`: bonded nucleotides in the starting structure. scadnano places helices 3 nm apart, so a
  well-placed crossover measures ~2.15 nm here. Anything listed in `worst_bonds` (> 2.6 nm) is a crossover at
  an offset where the backbones don't face each other. Move it.
- `metrics.bp_fraction_mean`: the fraction of designed base pairs that are hydrogen-bonded, averaged over the
  analysis window. **This is the main quality signal.**
  - ≥ 0.95: excellent. 0.85–0.95: acceptable, with local fraying.
  - < 0.85: significant problems. Check `low_occupancy_regions` (helix and offset ranges) and `bp_fraction_per_helix`.
- `rmsf_mean_nm`: thermal fluctuation about the mean structure. Compact bundles: ~0.5–1.5 nm. Larger values
  mean floppy regions.
- `rmsd_mean_structure_to_design_nm`: how far the relaxed mean structure is from the ideal design. Some
  deviation is normal because scadnano's starting geometry is idealized (helices 3 nm apart instead of ~2.5).
  Large values with low RMSF mean the structure is strained (twisted or bent).
- `extents_nm` vs `design_extents_nm`: principal dimensions of the relaxed and ideal structures.
- `shape_fidelity` (flat-shape tasks only): `iou` between the relaxed structure's outline, projected onto its
  best-fit plane, and the target shape. `flatness` is thickness divided by width (≈0.05–0.15 for a flat sheet;
  higher means the sheet curls or twists). A footprint IoU from validation that stays high while this IoU drops
  means the shape deforms in simulation: look for missing crossovers or twist.
- `WARNING: mock oxDNA runner`: the numbers are synthetic and come from a harness test mode. Rely on
  validation, not these metrics.

## Strategy

- Simulate once you have a valid, complete design. Use what the results show to make targeted fixes.
- Change one thing at a time between simulations, so you can tell what helped.
- Large designs (thousands of nucleotides) take much longer to simulate. Make your simulations count.
- Save each iteration as a new file (`designs/v2.sc`, `designs/v3.sc`, ...) and submit the best one,
  which isn't necessarily the latest.
