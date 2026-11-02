"""Model-agnostic deep agent for DNA origami design in scadnano.

This is the agent that ships: design skills and tools only, no simulation.
The internal benchmark (`synbio_bench`) runs this same agent, adds a
`submit_design` tool and a run budget, and tests submitted designs in oxDNA.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .tools import Workspace, make_tools

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SYSTEM_PROMPT = """\
You are an expert DNA nanotechnology engineer. You design DNA origami in scadnano (via its Python library).

Your environment
- Your file tools see a private workspace as `/`. Skills (read-only reference material) live under `/skills/`.
- Write scadnano design scripts with `write_file`, then execute them with `run_python`. Scripts run \
with the workspace root as working directory, so save designs with relative paths, e.g. `designs/v1.sc`.
- `find_crossover_offsets` tells you where crossovers between two helices are geometrically well placed.
- `check_design` runs a quick structural check of a saved design. Use it only when the user asks you to \
check or validate the design.

How to work
1. Read the relevant skills before writing code; they contain the scadnano API patterns and design rules you need.
2. Plan the design (lattice, helix layout, scaffold routing, staple crossovers, nicking) before coding.
3. Build the design, following the design rules, and save it as a .sc file.
4. Reply with a short summary: the design file's path, the scaffold routing, the crossover scheme and any \
open issues.

Work autonomously; there is no human to answer questions.
"""


def make_model(model: str, **kwargs: Any) -> BaseChatModel:
    """`model` is a LangChain `provider:name` string, e.g. `anthropic:claude-sonnet-5-5` or `openai:gpt-5.5`.

    `api_key_env` names an environment variable holding the key, which lets one
    provider class (e.g. OpenAI-compatible) point at several endpoints.
    """
    kwargs = dict(kwargs)
    if key_env := kwargs.pop("api_key_env", None):
        kwargs["api_key"] = os.environ[key_env]
    return init_chat_model(model, **kwargs)


def build_agent(
    model: BaseChatModel | str,
    ws: Workspace,
    skills_dir: Path = SKILLS_DIR,
    extra_tools: Sequence[Any] = (),
    system_prompt: str = SYSTEM_PROMPT,
    middleware: Sequence[Any] = (),
):
    backend = CompositeBackend(
        default=FilesystemBackend(root_dir=ws.root, virtual_mode=True),
        routes={"/skills/": FilesystemBackend(root_dir=skills_dir, virtual_mode=True)},
    )
    return create_deep_agent(
        model=model,
        tools=[*make_tools(ws), *extra_tools],
        system_prompt=system_prompt,
        skills=["/skills/"],
        backend=backend,
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        middleware=middleware,
        name="origami-designer",
    )
