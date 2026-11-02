"""The shipped agent (`origami_agent`) must not depend on oxDNA or the benchmark."""

import subprocess
import sys
from pathlib import Path

from origami_agent.agent import SKILLS_DIR, SYSTEM_PROMPT
from origami_agent.tools import Workspace, make_tools

PKG = SKILLS_DIR.parent


def test_shipped_package_never_imports_the_benchmark():
    code = (
        "import sys, origami_agent.agent, origami_agent.tools, origami_agent.geometry, origami_agent.design; "
        "print([m for m in sys.modules if m.startswith('synbio_bench')])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_shipped_code_has_no_benchmark_or_simulation_imports():
    for f in PKG.rglob("*.py"):
        imports = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.lstrip().startswith(("import ", "from "))]
        assert not any("synbio_bench" in ln for ln in imports), f
        assert not any("subprocess" in ln for ln in imports) or f.name == "tools.py", f  # only run_python starts processes


def test_shipped_tools_and_prompt_have_no_simulation_or_submission(tmp_path):
    names = {t.name for t in make_tools(Workspace(root=tmp_path))}
    assert names == {"run_python", "check_design", "find_crossover_offsets"}
    for word in ("simulate", "oxDNA", "submit", "budget"):
        assert word.lower() not in SYSTEM_PROMPT.lower(), word
    assert "only when the user asks" in SYSTEM_PROMPT


def test_run_python_handles_unicode_output(tmp_path):
    (tmp_path / "s.py").write_text('print("H0 → H1, 29 rows × 208 bp")', encoding="utf-8")
    run_python = next(t for t in make_tools(Workspace(root=tmp_path)) if t.name == "run_python")
    out = run_python.invoke({"script_path": "/s.py"})
    assert "exit code: 0" in out and "H0 → H1, 29 rows × 208 bp" in out


def test_shipped_skills_never_mention_simulation_or_grading():
    skills = list(SKILLS_DIR.rglob("*.md"))
    assert {p.parent.name for p in skills} == {"scadnano-python", "origami-design-rules"}
    for p in skills:
        text = p.read_text(encoding="utf-8").lower()
        for word in ("oxdna", "simulat", "submit_design", "validate_design", "iou", "grader"):
            assert word not in text, (Path(p).name, word)
