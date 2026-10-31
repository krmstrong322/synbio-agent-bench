"""Estimate the API cost of a run from its recorded token usage.

LangChain's `usage_metadata` reports `input_tokens` including cached tokens, with
the split in `input_token_details`:
- cache_read: tokens served from the provider's prompt cache
- cache_creation / ephemeral_5m_input_tokens / ephemeral_1h_input_tokens:
  tokens written to the cache (Anthropic)
Uncached input = input_tokens - cache reads - cache writes. Reasoning tokens are
part of `output_tokens` and billed as output.

Prices live in configs/models.yaml as `price_per_mtok` (USD per million tokens):
  {input, output, cache_read, cache_write_5m, cache_write_1h}
Missing cache prices fall back to the input price (a conservative over-estimate).
"""

from __future__ import annotations

from typing import Any


def estimate_cost(usage: dict[str, dict[str, Any]], price: dict[str, float] | None) -> dict[str, float] | None:
    """Cost in USD for a run's usage (keyed by model name), or None without prices."""
    if not price:
        return None
    totals = {"uncached_input": 0, "cache_read": 0, "cache_write_5m": 0, "cache_write_1h": 0, "output": 0}
    for u in usage.values():
        d = u.get("input_token_details") or {}
        read = d.get("cache_read") or 0
        w5 = d.get("ephemeral_5m_input_tokens") or 0
        w1 = d.get("ephemeral_1h_input_tokens") or 0
        if not (w5 or w1):  # providers that only report the aggregate
            w5 = d.get("cache_creation") or 0
        totals["cache_read"] += read
        totals["cache_write_5m"] += w5
        totals["cache_write_1h"] += w1
        totals["uncached_input"] += max(0, (u.get("input_tokens") or 0) - read - w5 - w1)
        totals["output"] += u.get("output_tokens") or 0
    rate = {
        "uncached_input": price["input"],
        "cache_read": price.get("cache_read", price["input"]),
        "cache_write_5m": price.get("cache_write_5m", price["input"]),
        "cache_write_1h": price.get("cache_write_1h", price["input"]),
        "output": price["output"],
    }
    parts = {k: totals[k] * rate[k] / 1e6 for k in totals}
    return {"total_usd": round(sum(parts.values()), 5), **{f"{k}_usd": round(v, 5) for k, v in parts.items()}}
