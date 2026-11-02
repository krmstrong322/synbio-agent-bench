"""LangChain tools the origami agent uses, bound to one workspace.

The agent's filesystem tools see the workspace as `/`. These tools accept the
same virtual paths ("/designs/x.sc" or "designs/x.sc") and resolve them inside
the workspace, refusing anything that escapes it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from .design import check_design, load_design
from .geometry import crossover_positions

MAX_OUTPUT_CHARS = 8000


@dataclass
class Workspace:
    root: Path
    script_timeout_s: int = 180

    def resolve(self, path: str) -> Path:
        p = (self.root / path.lstrip("/\\")).resolve()
        if p != self.root.resolve() and self.root.resolve() not in p.parents:
            raise ValueError(f"path {path!r} is outside the workspace")
        return p

    def virtual(self, p: Path) -> str:
        return "/" + p.resolve().relative_to(self.root.resolve()).as_posix()


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit // 2] + "\n...[truncated]...\n" + text[-limit // 2 :]


def _dump(obj: Any) -> str:
    return _clip(json.dumps(obj, indent=1, default=str))


def make_tools(ws: Workspace) -> list[BaseTool]:
    @tool
    def run_python(script_path: str) -> str:
        """Run a Python script from the workspace and return its exit code, stdout and stderr.

        Use this to execute scadnano design scripts you wrote with write_file. The
        script runs with the workspace root as its working directory, so write
        output files with relative paths (e.g. design.write_scadnano_file(filename="designs/v1.sc")).
        The `scadnano` and `numpy` packages are installed.
        """
        try:
            script = ws.resolve(script_path)
        except ValueError as e:
            return str(e)
        if not script.is_file():
            return f"no such file: {script_path}"
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=ws.root,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # scripts may print any Unicode, even on Windows
                timeout=ws.script_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"script timed out after {ws.script_timeout_s}s"
        return _clip(f"exit code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    @tool("check_design")
    def check_design_file(design_path: str) -> str:
        """Quick structural check of a .sc scadnano design. Use it only when the user asks you to check the design.

        Reports errors, warnings and a summary: illegal strands, missing scaffold,
        bases without sequence, non-complementary designed pairs, crossovers between
        helices that aren't neighbours, disconnected helix groups, plus helix count,
        base pairs per helix, scaffold length and coverage, staple lengths, and
        crossovers per helix pair. It does not simulate the structure.
        """
        try:
            return _dump(check_design(ws.resolve(design_path)))
        except ValueError as e:
            return str(e)

    @tool
    def find_crossover_offsets(design_path: str, helix: int, helix2: int) -> str:
        """List offsets where a crossover between two helices is geometrically well placed.

        Uses the helices' actual 3D layout (grid, positions, roll) in the given design.
        Results are given per direction of the strand on `helix`:
        - full_crossover_offsets: valid `offset` for design.add_full_crossover(helix, helix2, offset, forward)
        - half_crossover_offsets: valid `offset` for design.add_half_crossover(helix, helix2, offset, forward)
        Scaffold and staple strands on a helix run in opposite directions, so look up
        the direction of the strand you are connecting.
        """
        try:
            design = load_design(ws.resolve(design_path))
            return _dump(crossover_positions(design, helix, helix2))
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"

    return [run_python, check_design_file, find_crossover_offsets]
