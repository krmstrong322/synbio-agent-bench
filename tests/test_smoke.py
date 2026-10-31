"""End-to-end smoke tests that need neither API keys nor oxDNA (mock runner, scripted model)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from synbio_bench.origami.agent import build_agent
from synbio_bench.origami.design import validate_design
from synbio_bench.origami.oxdna import OxDNAConfig, simulate
from synbio_bench.origami.scoring import evaluate_submission
from synbio_bench.origami.tools import Workspace

ROOT = Path(__file__).resolve().parents[1]
HB6_SPEC = yaml.safe_load((ROOT / "tasks/origami/6hb-honeycomb.yaml").read_text())["spec"]


def build_reference(tmp_path: Path) -> Path:
    out = tmp_path / "hb6.sc"
    subprocess.run([sys.executable, "hb6_honeycomb.py", str(out)], cwd=ROOT / "reference_designs", check=True)
    return out


def test_reference_design_meets_spec(tmp_path):
    r = validate_design(build_reference(tmp_path), HB6_SPEC)
    assert r["ok"], r["errors"]
    assert all(v == 1.0 for v in r["checks"].values()), r["checks"]


def test_mock_simulation_reports_metrics(tmp_path):
    r = simulate(build_reference(tmp_path), tmp_path / "sim", OxDNAConfig(mode="mock"), protocol="quick")
    assert r["ok"], r
    m = r["metrics"]
    assert 0.9 < m["bp_fraction_mean"] <= 1.0
    assert r["initial_backbone"]["num_stretched_bonds"] == 0
    assert "WARNING" in r


class ScriptedModel(BaseChatModel):
    """Replays a fixed list of AI messages; enough to drive the agent loop without an API."""

    script: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(name: str, args: dict, i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}])


def test_agent_loop_with_scripted_model(tmp_path):
    script_src = (ROOT / "reference_designs/hb6_honeycomb.py").read_text()
    common_src = (ROOT / "reference_designs/common.py").read_text().replace(
        'Path(__file__).resolve().parents[1] / "src"', f'Path(r"{ROOT / "src"}")'
    )
    steps = [
        ("read_file", {"file_path": "/skills/scadnano-python/SKILL.md"}),
        ("write_file", {"file_path": "/common.py", "content": common_src}),
        ("write_file", {"file_path": "/make.py", "content": script_src}),
        ("run_python", {"script_path": "/make.py"}),  # writes hb6_honeycomb.sc in the workspace root
        ("validate_design_file", {"design_path": "/hb6_honeycomb.sc"}),
        ("find_crossover_offsets", {"design_path": "/hb6_honeycomb.sc", "helix": 0, "helix2": 1}),
        ("simulate_design", {"design_path": "/hb6_honeycomb.sc"}),
        ("submit_design", {"design_path": "/hb6_honeycomb.sc", "rationale": "reference routing"}),
    ]
    script = [_call(n, a, i) for i, (n, a) in enumerate(steps)] + [AIMessage(content="Done.")]
    ws = Workspace(root=tmp_path / "ws", spec=HB6_SPEC, oxdna=OxDNAConfig(mode="mock"))
    ws.root.mkdir()
    agent = build_agent(ScriptedModel(script=script), ws)
    state = agent.invoke({"messages": [{"role": "user", "content": "design a 6HB"}]}, config={"recursion_limit": 60})

    tool_msgs = {m.name: m.content for m in state["messages"] if m.type == "tool"}
    assert "scadnano Python API" in tool_msgs["read_file"]
    assert "exit code: 0" in tool_msgs["run_python"], tool_msgs["run_python"]
    assert json.loads(tool_msgs["validate_design_file"])["ok"]
    assert json.loads(tool_msgs["find_crossover_offsets"])["adjacent"] is True
    assert json.loads(tool_msgs["simulate_design"])["ok"]
    assert tool_msgs["submit_design"].startswith("Submitted.")
    assert ws.sim_count == 1 and ws.submitted and ws.submitted["valid"]

    grade = evaluate_submission(ws.root / "final" / "design.sc", HB6_SPEC, OxDNAConfig(mode="mock"), tmp_path / "eval")
    assert grade["valid"] and grade["overall"] > 0.9


def test_skills_are_read_only(tmp_path):
    ws = Workspace(root=tmp_path / "ws", spec=HB6_SPEC, oxdna=OxDNAConfig(mode="mock"))
    ws.root.mkdir()
    script = [_call("write_file", {"file_path": "/skills/evil.md", "content": "x"}, 0), AIMessage(content="ok")]
    agent = build_agent(ScriptedModel(script=script), ws)
    state = agent.invoke({"messages": [{"role": "user", "content": "x"}]}, config={"recursion_limit": 10})
    assert not (ROOT / "skills" / "evil.md").exists()
    assert any(m.type == "tool" and "denied" in str(m.content).lower() for m in state["messages"])


def test_tool_paths_cannot_escape_workspace(tmp_path):
    ws = Workspace(root=tmp_path / "ws", spec=HB6_SPEC, oxdna=OxDNAConfig(mode="mock"))
    ws.root.mkdir()
    try:
        ws.resolve("/../../etc/passwd")
    except ValueError:
        return
    raise AssertionError("path escaped the workspace")


def test_unwrap_rejoins_strands_split_across_periodic_boundary(tmp_path):
    import numpy as np

    from synbio_bench.origami.analysis import unwrap_strands

    # two 3-nt strands paired 0-5, 1-4, 2-3; strand 2 written one box length away
    box = np.array([10.0, 10.0, 10.0])
    pos = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1, 1, 0], [0.5, 1, 0], [0, 1, 0]], dtype=float)
    shifted = pos.copy()
    shifted[3:] += [10.0, -10.0, 0]
    strand = np.array([1, 1, 1, 2, 2, 2])
    out = unwrap_strands(shifted, strand, box, np.array([[0, 5]]))
    assert np.allclose(out, pos)
