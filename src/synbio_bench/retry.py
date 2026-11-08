"""Retry model calls that fail for transient provider reasons (rate limits, timeouts, overloads, 5xx).

These are infrastructure, not the model's doing: without a retry, one rate-limit blip ends the run with
`termination: error` (all four GPT-6.1 Sol runs did, on OpenRouter's "temporarily rate-limited upstream").
OpenRouter can report such errors inside the response body, which the OpenAI client raises as a plain
ValueError, so its own retries never see them; the message is matched too. Anything else (bad request, auth,
unknown model) is raised at once.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import ModelRetryMiddleware

TRANSIENT_TYPES = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                   "OverloadedError", "ServiceUnavailableError", "Timeout", "ReadTimeout", "ConnectTimeout"}
TRANSIENT_TEXT = re.compile(r"rate[- ]?limit|temporarily|overloaded|try again|retry shortly|timed? ?out|"
                            r"\b(429|500|502|503|504|529)\b|server error|upstream error|capacity", re.I)


def is_transient(exc: BaseException) -> bool:
    if not isinstance(exc, Exception):
        return False
    if type(exc).__name__ in TRANSIENT_TYPES:
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return bool(TRANSIENT_TEXT.search(str(exc)))


def api_retry_middleware(counter: dict[str, Any], max_retries: int = 6, initial_delay: float = 5.0,
                         max_delay: float = 60.0) -> ModelRetryMiddleware:
    """Retry transient errors with exponential backoff (5, 10, 20, 40, 60, 60 s); re-raise when they persist.

    `counter["api_retries"]` counts retries and `counter["last_retry_error"]` keeps the latest message.
    """
    counter.setdefault("api_retries", 0)

    def retry_on(exc: BaseException) -> bool:
        if is_transient(exc):
            counter["api_retries"] += 1
            counter["last_retry_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return True
        return False

    return ModelRetryMiddleware(max_retries=max_retries, retry_on=retry_on, on_failure="error",
                                initial_delay=initial_delay, max_delay=max_delay, backoff_factor=2.0)
