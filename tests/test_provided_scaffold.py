"""The provided-scaffold variant: a task supplies the scaffold sequence; the design must fold exactly it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from origami_agent.design import load_design
from synbio_bench.origami.spec import provided_scaffold_sequence, validate_design

ROOT = Path(__file__).resolve().parents[1]


def rectangle(tmp_path: Path) -> Path:
    out = tmp_path / "rect.sc"
    subprocess.run([sys.executable, "rothemund.py", "rect", str(out)], cwd=ROOT / "reference_designs",
                   check=True, capture_output=True)
    return out


def test_scaffold_matching_the_provided_sequence_passes(tmp_path):
    path = rectangle(tmp_path)
    seq = next(s for s in load_design(path).strands if s.is_scaffold).dna_sequence
    spec = {"scaffold": {"count": 1, "source": "provided", "sequence": seq}}
    v = validate_design(path, spec)
    assert v["checks"]["scaffold_is_provided"] == 1.0, v["errors"]

    # mRNA given with U must match the DNA scaffold assigned from it
    spec_rna = {"scaffold": {"count": 1, "source": "provided", "sequence": seq.replace("T", "U")}}
    assert validate_design(path, spec_rna)["checks"]["scaffold_is_provided"] == 1.0


def test_wrong_or_truncated_sequence_fails(tmp_path):
    path = rectangle(tmp_path)
    seq = next(s for s in load_design(path).strands if s.is_scaffold).dna_sequence
    spec = {"scaffold": {"count": 1, "source": "provided", "sequence": seq[:-50]}}  # 50 nt short
    v = validate_design(path, spec)
    assert v["checks"]["scaffold_is_provided"] == 0.0
    assert any("not the provided sequence" in e for e in v["errors"])


def test_sequence_from_file(tmp_path):
    (tmp_path / "mrna.fasta").write_text(">egfp\nACGUACGU\nACGU\n")  # FASTA header + U
    spec = {"scaffold": {"source": "provided", "sequence_file": "mrna.fasta"}}
    assert provided_scaffold_sequence(spec, base_dir=tmp_path) == "ACGTACGTACGT"


def test_run_agent_injects_the_sequence(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage

    from synbio_bench import runner
    from test_smoke import HB6_SPEC, ScriptedModel

    monkeypatch.setattr(runner, "make_model", lambda m, **kw: ScriptedModel(script=[AIMessage("done")]))
    spec = {**HB6_SPEC, "scaffold": {"source": "provided", "sequence": "ACGUACGU"}}
    runner.run_agent({"id": "t", "prompt": "go", "spec": spec}, {"id": "m", "model": "x:y"},
                     {"max_model_calls": 2, "last_chance_calls": 0}, tmp_path / "run", seed=0)
    injected = tmp_path / "run" / "workspace" / "inputs" / "scaffold.txt"
    assert injected.read_text().strip() == "ACGTACGT"  # U->T, available for the agent to read
