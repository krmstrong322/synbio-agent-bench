"""Per-run budget: model calls, wall time and a cost safety net.

The same rules apply to every model:
- As the run passes 50%, 75% and 90% of its budget, the next tool result gets a
  one-line notice, so the model can plan its remaining work.
- A reply cut off at the output token limit is dropped and the model is told to carry
  on (a few times per run at most), since that limit is ours, not the model's choice.
- When the budget runs out (or the model stops) without a submission, the runner
  sends one last-chance message. For a few more model calls only the tools needed
  to find, check and submit an existing design stay available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

NOTICE_THRESHOLDS = (0.5, 0.75, 0.9)
LAST_CHANCE_TOOLS = frozenset({"ls", "check_design", "submit_design"})

CUT_OFF_MESSAGE = """\
[harness] Your last reply hit the output token limit before it finished, so it was discarded. Carry on from \
where you were, and keep each reply shorter: plan briefly, and write long scripts in smaller pieces."""

LAST_CHANCE_MESSAGE = """\
[harness] The run has ended ({reason}) and no design has been submitted. This is your last chance: you \
have {calls} more model calls. Only `ls`, `check_design` and `submit_design` are available now: you cannot write or \
run code. Submit your best existing design file now; an unsubmitted design scores zero."""


@dataclass
class Budget:
    max_model_calls: int
    wall_timeout_s: float
    cost_cap_usd: float | None = None
    cost_fn: Callable[[], float | None] = lambda: None  # current run cost, set once usage is being tracked
    submitted_fn: Callable[[], bool] = lambda: False
    start: float = field(default_factory=time.time)
    model_calls: int = 0
    last_chance: bool = False
    notices: list[str] = field(default_factory=list)
    _announced: float = 0.0

    def elapsed(self) -> float:
        return time.time() - self.start

    def fractions(self) -> dict[str, float]:
        f = {"model_calls": self.model_calls / self.max_model_calls, "wall_time": self.elapsed() / self.wall_timeout_s}
        cost = self.cost_fn()
        if self.cost_cap_usd and cost is not None:
            f["cost"] = cost / self.cost_cap_usd
        return f

    def exceeded(self) -> str | None:
        """Why the main budget is used up (`step_limit`, `wall_timeout`, `cost_cap`), or None."""
        names = {"model_calls": "step_limit", "wall_time": "wall_timeout", "cost": "cost_cap"}
        for k, v in self.fractions().items():
            if v >= 1.0:
                return names[k]
        return None

    def notice(self) -> str | None:
        """A one-line budget notice when a new threshold has been crossed since the last one."""
        if self.last_chance:
            return None
        used = max(self.fractions().values())
        crossed = [t for t in NOTICE_THRESHOLDS if used >= t > self._announced]
        if not crossed:
            return None
        self._announced = crossed[-1]
        parts = [
            f"{self.model_calls} of {self.max_model_calls} model calls",
            f"{self.elapsed() / 60:.0f} of {self.wall_timeout_s / 60:.0f} min",
        ]
        if self.cost_cap_usd and (cost := self.cost_fn()) is not None:
            parts.append(f"${cost:.2f} of the ${self.cost_cap_usd:.2f} cost cap")
        status = "A design has been submitted." if self.submitted_fn() else "No design has been submitted yet."
        text = f"[harness] Budget {int(min(used, 1.0) * 100)}% used: {', '.join(parts)}. {status}"
        self.notices.append(text)
        return text

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "max_model_calls": self.max_model_calls,
            "wall_timeout_s": self.wall_timeout_s,
            "cost_cap_usd": self.cost_cap_usd,
            "notices": self.notices,
        }


class BudgetMiddleware(AgentMiddleware):
    """Counts model calls, appends budget notices to tool results, and narrows tools in the last chance."""

    def __init__(self, budget: Budget):
        super().__init__()
        self.budget = budget

    def wrap_model_call(self, request, handler):
        self.budget.model_calls += 1
        if self.budget.last_chance:
            request = request.override(tools=[t for t in request.tools if _tool_name(t) in LAST_CHANCE_TOOLS])
        return handler(request)

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        if isinstance(result, ToolMessage) and (text := self.budget.notice()):
            if isinstance(result.content, str):
                result.content = f"{result.content}\n\n{text}"
            else:
                result.content = [*result.content, {"type": "text", "text": text}]
        return result


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        return tool.get("name") or tool.get("function", {}).get("name", "")
    return tool.name
