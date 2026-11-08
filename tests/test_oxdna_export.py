"""scadnano's oxDNA export crashes on helices with min_offset > 0; origami_agent.design.oxdna_format works around it."""

from __future__ import annotations

import pytest
import scadnano as sc

from origami_agent.design import oxdna_format
from origami_agent.geometry import crossover_positions
from synbio_bench.origami.oxdna import export_design


def two_helices(min_offset: int, deletion: bool = False) -> sc.Design:
    helices = [sc.Helix(max_offset=48, min_offset=min_offset, idx=i) for i in range(2)]
    d = sc.Design(helices=helices, grid=sc.square)
    d.draw_strand(0, 16).move(32).cross(1).move(-32).with_sequence("ACGT" * 16, assign_complement=False)
    if deletion:
        d.add_deletion(0, 20)
    return d


def test_scadnano_bug_still_present():
    """If this starts failing, scadnano fixed its export and the workaround can go."""
    with pytest.raises(IndexError):
        two_helices(8).to_oxdna_format(warn_duplicate_strand_names=False)


def test_min_offset_above_zero_exports_same_coordinates():
    assert oxdna_format(two_helices(8)) == oxdna_format(two_helices(0))
    d = two_helices(8)
    oxdna_format(d)
    assert d.helices[0].min_offset == 8  # the caller's design is not changed


def test_negative_min_offset_with_deletions_is_refused():
    with pytest.raises(ValueError, match="min_offset"):
        oxdna_format(two_helices(-8, deletion=True))
    oxdna_format(two_helices(-8))  # without insertions/deletions it is exported as is


def test_export_and_agent_geometry_handle_min_offset(tmp_path):
    d = two_helices(8)
    path = tmp_path / "d.sc"
    d.write_scadnano_file(filename=path.name, directory=str(tmp_path))
    assert export_design(path, tmp_path / "out")["n_nucleotides"] == 64
    assert crossover_positions(d, 0, 1)  # the agent's crossover finder used to crash here too
