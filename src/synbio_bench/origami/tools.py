"""LangChain tools the origami agent uses, bound to one run's workspace.

The agent's filesystem tools see the workspace as `/`. These tools accept the
same virtual paths ("/designs/x.sc" or "designs/x.sc") and resolve them inside
the workspace, refusing anything that escapes it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from .design import load_design, validate_design
from .geometry import crossover_positions
from .oxdna import OxDNAConfig, simulate

MAX_OUTPUT_CHARS = 8000


@dataclass
class Workspace:
    root: Path
    spec: dict[str, Any]
    oxdna: OxDNAConfig
    max_simulations: int = 4
    script_timeout_s: int = 180
    sim_count: int = 0
    submitted: dict[str, Any] | None = None

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
                text=True,
                timeout=ws.script_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"script timed out after {ws.script_timeout_s}s"
        return _clip(f"exit code: {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")

    @tool
    def validate_design_file(design_path: str) -> str:
        """Load a .sc scadnano design, check it against the task specification and summarise it.

        Reports errors (must fix), warnings, per-check pass fractions, and a summary:
        helix count and grid positions, base pairs per helix, scaffold length and
        coverage, staple length histogram, crossovers per helix pair, mismatches.
        """
        try:
            return _dump(validate_design(ws.resolve(design_path), ws.spec))
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

    @tool
    def simulate_design(design_path: str) -> str:
        """Simulate a design in oxDNA (relax, then short production MD) and report structural metrics.

        Returns base-pair occupancy (fraction of designed pairs formed, overall and
        per helix, plus the worst regions), RMSF, RMSD of the mean structure to the
        ideal design, extents, and long backbone bonds in the starting structure
        (which usually mean badly placed crossovers). Simulations are expensive and
        limited per run; validate the design first.
        """
        if ws.sim_count >= ws.max_simulations:
            return f"simulation budget exhausted ({ws.max_simulations} used). Submit your best design."
        try:
            path = ws.resolve(design_path)
        except ValueError as e:
            return str(e)
        check = validate_design(path, ws.spec)
        hard = [e for e in check["errors"] if e.startswith(("could not load", "illegal", "two nucleotides"))]
        if hard or not check.get("summary"):
            return "design cannot be simulated:\n" + "\n".join(check["errors"])
        ws.sim_count += 1
        out = ws.root / "sims" / f"sim_{ws.sim_count:02d}"
        result = simulate(path, out, ws.oxdna, protocol="quick", seed=ws.sim_count, shape=ws.spec.get("shape"))
        result["work_dir"] = ws.virtual(out)
        result["simulations_remaining"] = ws.max_simulations - ws.sim_count
        return _dump(result)

    @tool
    def submit_design(design_path: str, rationale: str) -> str:
        """Submit a design as your final answer. You may resubmit; the last submission counts.

        `rationale` should briefly explain the scaffold routing, crossover scheme and
        any changes made in response to simulation results.
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
        (final / "rationale.md").write_text(rationale)
        check = validate_design(final / "design.sc", ws.spec)
        ws.submitted = {"source": ws.virtual(path), "valid": check["ok"], "errors": check["errors"]}
        msg = "Submitted." if check["ok"] else "Submitted, but the design still has errors:\n" + "\n".join(check["errors"])
        return msg + " If you are finished, reply with a short final summary."

    return [run_python, validate_design_file, find_crossover_offsets, simulate_design, submit_design]
