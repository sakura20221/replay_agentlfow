#!/usr/bin/env python3
"""Reproduce a ScEnsemble call standalone and show the real prompt and reply.

The ValidationError counted during runs was diagnosed by inference -- "the XML
field extraction must be failing" -- without anyone looking at what the model was
actually sent or what it answered. This drives the repo's own machinery
(ActionNode.from_pydantic(...).fill(mode="xml_fill")) so the request is identical
to a real run, and wraps llm.aask to capture the exact prompt and raw reply.

    cd third_party/maas && PYTHONPATH=. python ../../shims/maas_family/probe_scensemble.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from maas.actions.action_node import ActionNode  # noqa: E402
from maas.configs.models_config import ModelsConfig  # noqa: E402
from maas.ext.maas.scripts.optimized.MATH.train.template.op_prompt import (  # noqa: E402
    SC_ENSEMBLE_PROMPT,
)
from maas.ext.maas.scripts.optimized.MATH.train.template.operator_an import (  # noqa: E402
    ScEnsembleOp,
)
from maas.llm import LLM  # noqa: E402

SHARED = Path.cwd().parents[1] / "shared"
sys.path.insert(0, str(SHARED))
import bench as shared_bench  # noqa: E402

CAPTURED: list[tuple[str, str]] = []


def instrument(llm) -> None:
    original = llm.aask

    async def wrapped(msg, *args, **kwargs):
        reply = await original(msg, *args, **kwargs)
        CAPTURED.append((str(msg), str(reply)))
        return reply

    llm.aask = wrapped


async def main() -> None:
    row = shared_bench.load("math")[0]
    problem = shared_bench.question_text("math", row)

    solutions = [
        "Converting to polar: r = 3, theta = pi/2. The answer is (3, pi/2).",
        "Using x = r cos t and y = r sin t gives r = 3 and t = pi/2.",
        "r = sqrt(0^2 + 3^2) = 3, theta = pi/2 since the point is on the +y axis.",
    ]
    mapping = {chr(65 + i): i for i in range(len(solutions))}
    solution_text = "".join(f"{chr(65 + i)}: \n{s}\n\n\n" for i, s in enumerate(solutions))

    # The repo builds this with str.format(); use plain replacement so the
    # LaTeX braces in the problem cannot raise here.
    prompt = SC_ENSEMBLE_PROMPT.replace("{problem}", problem).replace("{solutions}", solution_text)

    llm = LLM(ModelsConfig.default().get("qwen3-8b"))
    instrument(llm)

    print(f"letter mapping: {mapping}")
    print(f"ScEnsembleOp fields: {list(ScEnsembleOp.model_fields)}")

    error = None
    result = None
    try:
        node = await ActionNode.from_pydantic(ScEnsembleOp).fill(
            context=prompt, llm=llm, mode="xml_fill"
        )
        result = node.instruct_content.model_dump()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    for index, (sent, reply) in enumerate(CAPTURED):
        print("\n" + "=" * 72)
        print(f"CALL {index}: EXACT PROMPT SENT ({len(sent)} chars)")
        print("=" * 72)
        print(sent[-2200:] if len(sent) > 2200 else sent)
        print("=" * 72)
        print(f"CALL {index}: RAW REPLY ({len(reply)} chars)")
        print("=" * 72)
        print(reply[:1800])
        print("=" * 72)
        found = re.findall(r"<solution_letter>(.*?)</solution_letter>", reply, re.DOTALL)
        print(f"  <solution_letter> present : {bool(found)}"
              + (f"  value={found[-1].strip()!r}" if found else ""))
        letters = [c for c in re.findall(r"\b([A-Z])\b", reply) if c in mapping]
        print(f"  valid bare letters in text: {letters[:6]}")

    print("\n" + "-" * 72)
    print(f"fill() result: {result}")
    print(f"fill() error : {error}")


asyncio.run(main())
