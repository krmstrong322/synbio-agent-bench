"""Look up OpenRouter models and write checked entries for configs/models.yaml.

The agent needs tool calling, so a model without it can't run the benchmark. This script reads OpenRouter's
public model list (no API key needed) and, for each model:
- checks the slug exists and the model supports `tools`,
- takes its current prices for cost tracking (and warns when OpenRouter prices it by time of day),
- caps max_tokens at what the model's provider allows,
- prints a models.yaml entry, or appends it with --add.

    uv run python scripts/openrouter_models.py --search qwen            # find slugs (tool-capable first)
    uv run python scripts/openrouter_models.py openai/gpt-5.5 qwen/qwen4-max
    uv run python scripts/openrouter_models.py openai/gpt-5.5 qwen/qwen4-max --add
    uv run python scripts/openrouter_models.py z-ai/glm-6 --reasoning-effort high --add

Runs go through `OPENROUTER_API_KEY` from .env. Prices are what OpenRouter lists today: re-run before a big
experiment and compare the cost in agent.json against OpenRouter's activity page.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS_YAML = ROOT / "configs" / "models.yaml"
API = "https://openrouter.ai/api/v1/models"
MAX_TOKENS = 64000  # same output budget as the Claude runs (reasoning tokens count toward it)


def fetch_models() -> list[dict[str, Any]]:
    with urllib.request.urlopen(API, timeout=30) as r:
        return json.load(r)["data"]


def per_mtok(value: Any) -> float | None:
    """OpenRouter prices are USD per token, as strings; return USD per million tokens."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v * 1e6, 6) if v >= 0 else None


def model_id(slug: str) -> str:
    """Short run id from a slug: 'qwen/qwen4-max' -> 'qwen4-max' (dates and ':free' style suffixes dropped)."""
    name = slug.split("/", 1)[-1].split(":", 1)[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()


def entry_for(m: dict[str, Any], reasoning_effort: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """models.yaml entry for an OpenRouter model, plus warnings."""
    warnings: list[str] = []
    params = set(m.get("supported_parameters") or [])
    if "tools" not in params:
        warnings.append("does NOT support tool calling: the agent cannot run on it")
    p = m.get("pricing") or {}
    price = {"input": per_mtok(p.get("prompt")), "output": per_mtok(p.get("completion"))}
    if (cr := per_mtok(p.get("input_cache_read"))) is not None:
        price["cache_read"] = cr
    if (cw := per_mtok(p.get("input_cache_write"))) is not None:
        price["cache_write_5m"] = cw
    if price["input"] is None or price["output"] is None:
        warnings.append("no usable price listed; cost estimates will be missing")
        price = {k: v for k, v in price.items() if v is not None}
    for o in p.get("overrides") or []:
        if o.get("min_prompt_tokens"):
            # Long-prompt tier: every call whose prompt reaches min_prompt_tokens is billed at these rates.
            # cost.py prices runs from totals and can't apply it; scripts/estimate_costs.py does, per call.
            tier = {"min_prompt_tokens": int(o["min_prompt_tokens"]), "input": per_mtok(o.get("prompt")),
                    "output": per_mtok(o.get("completion"))}
            if (v := per_mtok(o.get("input_cache_read"))) is not None:
                tier["cache_read"] = v
            if (v := per_mtok(o.get("input_cache_write"))) is not None:
                tier["cache_write_5m"] = v
            price["long_prompt"] = tier
            warnings.append(f"calls with prompts of {tier['min_prompt_tokens']:,}+ tokens cost more "
                            f"(input ${tier['input']}/M, output ${tier['output']}/M); recorded as long_prompt")
        elif o.get("utc_days") or "utc_start" in o:
            if not any("time of day" in w for w in warnings):
                warnings.append("priced by time of day/week on OpenRouter; the entry uses the base rate, so "
                                "estimates for runs in peak hours will be low")
        elif not any("price override" in w for w in warnings):
            warnings.append(f"has a price override this script doesn't understand: {o}")
    if price.get("input") == 0 and price.get("output") == 0:
        warnings.append("free variant: expect tight rate limits")
    top = m.get("top_provider") or {}
    max_tokens = MAX_TOKENS
    if top.get("max_completion_tokens"):
        max_tokens = min(MAX_TOKENS, int(top["max_completion_tokens"]))
        if max_tokens < MAX_TOKENS:
            warnings.append(f"max output is {max_tokens} tokens (Claude runs had {MAX_TOKENS})")
    kwargs: dict[str, Any] = {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
                              "use_responses_api": False, "max_tokens": max_tokens}
    if reasoning_effort:
        if "reasoning" in params:
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
        else:
            warnings.append("reasoning effort requested but OpenRouter lists no reasoning control for it")
    entry = {"id": model_id(m["id"]), "model": f"openai:{m['id']}", "kwargs": kwargs, "price_per_mtok": price}
    return entry, warnings


def entry_yaml(entry: dict[str, Any], comment: str) -> str:
    """One list item in the flow style models.yaml already uses."""
    def flow(d: dict[str, Any]) -> str:
        return yaml.safe_dump(d, default_flow_style=True, sort_keys=False, width=10_000).strip()

    lines = [f"  - id: {entry['id']}", f"    model: {entry['model']}",
             f"    kwargs: {flow(entry['kwargs'])}", f"    price_per_mtok: {flow(entry['price_per_mtok'])}"]
    return f"  # {comment}\n" + "\n".join(lines) + "\n"


def existing(path: Path) -> dict[str, str]:
    """model string -> id for the entries already in models.yaml."""
    return {m["model"]: m["id"] for m in (yaml.safe_load(path.read_text()) or {}).get("models", [])}


def unique_id(entry_id: str, taken: set[str]) -> str:
    """An id not used yet: a direct-API entry may already own the plain name (e.g. gpt-5.5 via OpenAI)."""
    if entry_id not in taken:
        return entry_id
    k, candidate = 2, f"{entry_id}-or"
    while candidate in taken:
        candidate, k = f"{entry_id}-or{k}", k + 1
    return candidate


def search(models: list[dict[str, Any]], term: str) -> None:
    term = term.lower()
    hits = [m for m in models if term in m["id"].lower() or term in (m.get("name") or "").lower()]
    hits.sort(key=lambda m: ("tools" not in (m.get("supported_parameters") or []), -(m.get("created") or 0)))
    print(f"{'slug':<52} {'tools':<5} {'context':>9} {'in $/M':>8} {'out $/M':>8}  released")
    for m in hits:
        p = m.get("pricing") or {}
        created = dt.datetime.fromtimestamp(m.get("created") or 0, dt.timezone.utc).date()
        tools = "yes" if "tools" in (m.get("supported_parameters") or []) else "NO"
        print(f"{m['id']:<52} {tools:<5} {m.get('context_length') or 0:>9} {per_mtok(p.get('prompt')) or 0:>8.3f} "
              f"{per_mtok(p.get('completion')) or 0:>8.3f}  {created}")
    if not hits:
        print(f"no OpenRouter model matches {term!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slugs", nargs="*", help="OpenRouter model slugs, e.g. openai/gpt-5.5")
    p.add_argument("--search", help="list models whose slug or name contains this")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                   help="ask OpenRouter for this reasoning effort (default: the model's own default)")
    p.add_argument("--add", action="store_true", help="append the entries to configs/models.yaml")
    args = p.parse_args()

    models = fetch_models()
    if args.search:
        search(models, args.search)
    if not args.slugs:
        return
    by_slug = {m["id"]: m for m in models} | {m.get("canonical_slug"): m for m in models if m.get("canonical_slug")}
    have = existing(MODELS_YAML)
    taken = set(have.values())
    today = dt.date.today().isoformat()
    out, ids, bad = [], [], False
    for slug in args.slugs:
        m = by_slug.get(slug)
        if not m:
            print(f"[miss] {slug}: not on OpenRouter (try --search {slug.split('/')[-1][:12]})", file=sys.stderr)
            bad = True
            continue
        entry, warnings = entry_for(m, args.reasoning_effort)
        for w in warnings:
            print(f"[warn] {slug}: {w}", file=sys.stderr)
        if any("tool calling" in w for w in warnings):
            bad = True
            continue
        if entry["model"] in have:
            print(f"[skip] {slug}: already in {MODELS_YAML.name} as id {have[entry['model']]!r}", file=sys.stderr)
            ids.append(have[entry["model"]])
            continue
        if entry["id"] in taken:
            new = unique_id(entry["id"], taken)
            print(f"[note] {slug}: id {entry['id']!r} is taken by another entry; using {new!r}", file=sys.stderr)
            entry["id"] = new
        taken.add(entry["id"])
        out.append(entry_yaml(entry, f"{m.get('name') or slug} via OpenRouter; prices as of {today}"))
        ids.append(entry["id"])
    text = "".join(out)
    if text:
        print(text, end="")
    if args.add and text:
        with open(MODELS_YAML, "a", encoding="utf-8") as f:
            f.write(text)
        yaml.safe_load(MODELS_YAML.read_text())  # fail loudly if the file no longer parses
        print(f"[add ] appended {len(out)} model(s) to {MODELS_YAML}", file=sys.stderr)
    if ids:
        print(f"\nmodels: [{', '.join(ids)}]   # for the experiment config", file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
