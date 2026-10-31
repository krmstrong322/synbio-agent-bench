"""HTML report: summary table plus a card per run with design and structure figures.

Images are linked relatively (the report lives in runs/<experiment>/), or
embedded as data URIs with `embed=True` so the file can be shared on its own.
"""

from __future__ import annotations

import base64
import json
import traceback
from html import escape
from pathlib import Path
from typing import Any

import yaml

from .origami.viz import figures_for_run

ROOT = Path(__file__).resolve().parents[2]
OXVIEW_URL = "https://sulcgroup.github.io/oxdna-viewer/"

CSS = """
:root { --fg:#1d232a; --muted:#5f6b77; --line:#dde2e7; --card:#ffffff; --bg:#f6f7f9; --ok:#2f8f46; --bad:#c0392b; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; font:14px/1.5 system-ui, sans-serif; color:var(--fg); background:var(--bg); }
main { max-width:1240px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:36px 0 12px; } h3 { font-size:15px; margin:0; }
.sub { color:var(--muted); margin:0 0 20px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }
th { background:#eef1f4; } .scroll { overflow-x:auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:12px 0; }
.head { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:baseline; }
.badge { font-size:12px; padding:1px 8px; border-radius:10px; background:#eef1f4; color:var(--muted); }
.badge.ok { background:#e3f3e7; color:var(--ok); } .badge.bad { background:#fbe7e5; color:var(--bad); }
.metrics { display:flex; flex-wrap:wrap; gap:4px 18px; margin:8px 0; color:var(--muted); font-size:13px; }
.metrics b { color:var(--fg); font-weight:600; }
.fig { margin:10px 0; overflow-x:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
.fig img { display:block; max-width:none; }
.fig img.fit { max-width:100%; height:auto; }
details { margin:6px 0; } summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; font-size:12px; background:#f3f5f7; padding:8px; border-radius:6px; }
ul.errors { color:var(--bad); margin:6px 0; padding-left:18px; }
a { color:#2c5aa0; }
"""


def _src(run_dir: Path, rel: str, root: Path, embed: bool) -> str:
    path = run_dir / rel
    if not embed:
        return escape(path.relative_to(root).as_posix())
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _fmt(v: Any, digits: int = 2) -> str:
    return "–" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else escape(str(v)))


def _card(run_dir: Path, info: dict, root: Path, embed: bool, figures: bool) -> str:
    g = info.get("grade", {})
    t = info.get("tokens", {})
    rel = run_dir.relative_to(root).as_posix()
    term = info.get("termination", "?")
    valid = g.get("valid")
    head = (
        f'<div class="head"><h3>{escape(info["model"])} · {escape(run_dir.name)}</h3>'
        f'<span class="badge {"ok" if term in ("completed", "reference") else "bad"}">{escape(term)}</span>'
        f'<span class="badge {"ok" if valid else "bad"}">{"valid" if valid else "invalid"}</span>'
        f'<span>overall <b>{_fmt(g.get("overall"))}</b></span></div>'
    )
    metrics = [
        ("spec", g.get("spec_score")), ("bp formed", g.get("bp_fraction_mean")), ("shape IoU", g.get("shape_score")),
        ("RMSF nm", g.get("rmsf_mean_nm")), ("sims used", info.get("simulations_used")),
        ("tokens in/out k", f'{(t.get("input_tokens") or 0) / 1000:.0f} / {(t.get("output_tokens") or 0) / 1000:.0f}' if t else None),
        ("time min", (info.get("wall_time_s") or 0) / 60 if info.get("wall_time_s") else None),
    ]
    body = [head, '<div class="metrics">' + "".join(f"<span>{k} <b>{_fmt(v, 1 if k == 'time min' else 2)}</b></span>" for k, v in metrics if v is not None) + "</div>"]
    if g.get("errors"):
        body.append('<ul class="errors">' + "".join(f"<li>{escape(e)}</li>" for e in g["errors"][:4]) + "</ul>")

    if figures:
        try:
            spec = yaml.safe_load((ROOT / "tasks" / f"{info['task']}.yaml").read_text())["spec"]
            figs = figures_for_run(run_dir, spec, f"{info['task']} · {info['model']} · {run_dir.name}")
        except Exception as e:  # noqa: BLE001 - a broken figure must not break the report
            figs = {}
            body.append(f"<details><summary>figure error</summary><pre>{escape(traceback.format_exc()[-1500:])}</pre></details>")
        if "design" in figs:
            body.append(f'<div class="fig"><img src="{_src(run_dir, figs["design"], root, embed)}" alt="design diagram"></div>')
        if "structure_occupancy" in figs:
            body.append(f'<div class="fig"><img class="fit" src="{_src(run_dir, figs["structure_occupancy"], root, embed)}" alt="simulated structure, base-pair occupancy"></div>')
        if "structure_strands" in figs:
            body.append(f'<details><summary>structure coloured by strand</summary><div class="fig"><img class="fit" src="{_src(run_dir, figs["structure_strands"], root, embed)}" alt="simulated structure by strand"></div></details>')
        if "sim_dir" in figs:
            sd = f'{rel}/{figs["sim_dir"]}'
            body.append(
                f'<details><summary>open in oxView</summary><p>Drag <a href="{escape(sd)}/initial.top">initial.top</a> and '
                f'<a href="{escape(sd)}/last_conf.dat">last_conf.dat</a> into <a href="{OXVIEW_URL}">oxView</a>.</p></details>'
            )

    rationale = run_dir / "workspace" / "final" / "rationale.md"
    if rationale.exists():
        body.append(f"<details><summary>agent's rationale</summary><pre>{escape(rationale.read_text(encoding='utf-8', errors='replace')[:4000])}</pre></details>")
    links = [(n, f"{rel}/{p}") for n, p in (("design.sc", "workspace/final/design.sc"), ("transcript", "transcript.json"), ("result", "result.json")) if (run_dir / p).exists()]
    if links:
        body.append('<div class="metrics">' + " · ".join(f'<a href="{escape(h)}">{escape(n)}</a>' for n, h in links) + "</div>")
    return '<div class="card">' + "\n".join(body) + "</div>"


def write_html(root: Path, exp: dict, summary_lines: list[str], embed: bool = False, figures: bool = True) -> Path:
    results = sorted(root.glob("*/*/rep*/result.json"))
    by_task: dict[str, list[tuple[Path, dict]]] = {}
    for p in results:
        info = json.loads(p.read_text())
        by_task.setdefault(info["task"], []).append((p.parent, info))

    table = [ln for ln in summary_lines if ln.startswith("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in table if not set(ln) <= set("|-")]
    html_table = "<div class=\"scroll\"><table>" + "".join(
        "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{escape(c)}</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>"
        for i, r in enumerate(rows)
    ) + "</table></div>"

    sections = []
    for task, runs in sorted(by_task.items()):
        cards = [_card(d, info, root, embed, figures) for d, info in sorted(runs, key=lambda x: (x[1]["model"], x[0].name))]
        sections.append(f"<h2>{escape(task)}</h2>" + "\n".join(cards))

    mock = any(json.loads(p.read_text()).get("grade", {}).get("sim_runner") == "mock" for p in results)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(exp['name'])} report</title><style>{CSS}</style></head>
<body><main>
<h1>{escape(exp['name'])}</h1>
<p class="sub">{len(results)} graded runs · design diagrams show failed base pairs in red · structures are oxDNA mean structures</p>
{'<p class="sub"><b>Warning:</b> some runs used the mock oxDNA runner; their simulation metrics are synthetic.</p>' if mock else ''}
{html_table}
{''.join(sections)}
</main></body></html>
"""
    out = root / ("report_embedded.html" if embed else "report.html")
    out.write_text(html, encoding="utf-8")
    return out
