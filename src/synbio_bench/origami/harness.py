"""What the benchmark adds to the shipped agent: a way to hand in the answer, and the run budget.

The agent itself is `origami_agent` unchanged (same prompt, tools and skills), so scores measure what
ships. oxDNA never runs during a run; it is used only to grade the submitted design afterwards.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, tool

from origami_agent.agent import SYSTEM_PROMPT as AGENT_PROMPT
from origami_agent.tools import Workspace

from .spec import validate_design

BENCHMARK_NOTE = """
Benchmark run
- When your design is finished, record it with `submit_design`: that file is what gets evaluated, and a design \
that is never submitted scores zero. You may resubmit; the last submission counts.
- The run has a budget of model calls and time. Tool results carry a notice at 50%, 75% and 90% of it; \
submit a finished design early and improve it afterwards rather than risk running out.
"""

SYSTEM_PROMPT = AGENT_PROMPT + BENCHMARK_NOTE


@dataclass
class Submission:
    spec: dict[str, Any]
    record: dict[str, Any] | None = None  # the last submission, with the grader's spec check (not shown to the agent)


def make_submit_tool(ws: Workspace, sub: Submission) -> BaseTool:
    @tool
    def submit_design(design_path: str, rationale: str) -> str:
        """Submit a design as your final answer. You may resubmit; the last submission counts.

        `rationale` should briefly explain the scaffold routing and crossover scheme.
        """
        try:
            path = ws.resolve(design_path)
        except ValueError as e:
            return str(e)
        if not path.is_file():
            return f"no such file: {design_path}"
        final = ws.root / "final"
        final.mkdir(exist_ok=True)
        shutil.copyfile(path, final / "design.sc")
        (final / "rationale.md").write_text(rationale, encoding="utf-8")
        try:
            check = validate_design(final / "design.sc", sub.spec)
        except Exception as e:  # noqa: BLE001 - a grader bug must never cost the model its submission
            check = {"ok": None, "errors": [f"spec check failed: {type(e).__name__}: {e}"]}
        sub.record = {"source": ws.virtual(path), "valid": check["ok"], "errors": check["errors"]}
        return "Submitted. If you are finished, reply with a short final summary."

    return submit_design
