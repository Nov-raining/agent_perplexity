from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .agent import build_agent, format_report, format_report_json
from .rag import add_memory, add_to_vector_store, load_memory, retrieve_memory, retrieve_similar


def _select_extra_keywords(mode: str, base_keywords: List[str]) -> List[str]:
    extras: List[str] = []
    memories = retrieve_memory(base_keywords, top_k=2)
    for item in memories:
        for key in item.get("keywords", []):
            if key not in extras and key not in base_keywords:
                extras.append(key)
    if not extras and mode == "real":
        extras.extend(["原始出处", "首发"])
    return extras


def _select_extra_keywords_vector(base_keywords: List[str]) -> List[str]:
    items = load_memory().get("items", [])
    query = " ".join([k for k in base_keywords if k])
    indices = retrieve_similar(query, top_k=2)
    extras: List[str] = []
    for idx in indices:
        if idx < 0 or idx >= len(items):
            continue
        item = items[idx]
        for key in item.get("keywords", []):
            if key not in extras and key not in base_keywords:
                extras.append(key)
    return extras


def build_chain(mode: str, output: str):
    try:
        from langchain_core.runnables import RunnableLambda, RunnableSequence
    except Exception as exc:
        raise RuntimeError(
            "langchain-core not installed. Install with 'pip install langchain-core'"
        ) from exc

    def _step_perceive(payload: Dict[str, str]) -> Dict[str, str]:
        image_path = payload.get("image_path", "")
        agent = build_agent(mode)
        perception = agent.tools.vlm.perceive(image_path)
        keywords = [perception.subject_type, perception.region_hint] + perception.features[:2]
        payload["base_keywords"] = "|".join([k for k in keywords if k])
        payload["perception"] = perception
        return payload

    def _step_route(payload: Dict[str, str]) -> Dict[str, str]:
        base_keywords = payload.get("base_keywords", "").split("|") if payload.get("base_keywords") else []
        extra = _select_extra_keywords_vector(base_keywords)
        if not extra:
            extra = _select_extra_keywords(mode, base_keywords)
        payload["extra_keywords"] = "|".join(extra)
        return payload

    def _step_run(payload: Dict[str, str]) -> str:
        image_path = payload.get("image_path", "")
        base_keywords = payload.get("base_keywords", "").split("|") if payload.get("base_keywords") else []
        extra_keywords = payload.get("extra_keywords", "").split("|") if payload.get("extra_keywords") else []
        agent = build_agent(mode)
        result = agent.run(image_path, extra_keywords=extra_keywords)
        if result.reports:
            last = result.reports[-1]
            memory_item = {
                "keywords": base_keywords + extra_keywords,
                "conclusion": last.conclusion.conclusion,
                "source_url": last.osint.source_url,
            }
            add_memory(memory_item["keywords"], memory_item["conclusion"], memory_item["source_url"])
            add_to_vector_store(memory_item)
        if output == "json":
            return format_report_json(result)
        return format_report(result)

    return RunnableSequence(
        RunnableLambda(_step_perceive),
        RunnableLambda(_step_route),
        RunnableLambda(_step_run),
    )


def run_with_langchain(image_path: str, mode: str, output: str) -> str:
    chain = build_chain(mode, output)
    return chain.invoke({"image_path": image_path})
