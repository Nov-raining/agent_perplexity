from __future__ import annotations

import json
from typing import Dict

from .agent import build_agent, format_report, format_report_json


def build_chain(mode: str, output: str):
    try:
        from langchain_core.runnables import RunnableLambda
    except Exception as exc:
        raise RuntimeError(
            "langchain-core not installed. Install with 'pip install langchain-core'"
        ) from exc

    def _run(payload: Dict[str, str]) -> str:
        image_path = payload.get("image_path", "")
        agent = build_agent(mode)
        result = agent.run(image_path)
        if output == "json":
            return format_report_json(result)
        return format_report(result)

    return RunnableLambda(_run)


def run_with_langchain(image_path: str, mode: str, output: str) -> str:
    chain = build_chain(mode, output)
    return chain.invoke({"image_path": image_path})
