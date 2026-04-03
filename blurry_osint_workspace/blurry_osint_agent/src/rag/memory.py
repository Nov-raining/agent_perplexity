from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def _rag_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "rag")
    os.makedirs(base, exist_ok=True)
    return base


def memory_path() -> str:
    return os.path.join(_rag_dir(), "memory.json")


def load_memory() -> Dict[str, Any]:
    path = memory_path()
    if not os.path.exists(path):
        return {"items": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"items": []}


def save_memory(data: Dict[str, Any]) -> None:
    with open(memory_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def add_memory(keywords: List[str], conclusion: str, source_url: str) -> None:
    data = load_memory()
    items = data.get("items", [])
    items.append(
        {
            "keywords": [k for k in keywords if k],
            "conclusion": conclusion,
            "source_url": source_url,
        }
    )
    data["items"] = items[-500:]
    save_memory(data)


def retrieve_memory(keywords: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    data = load_memory()
    items = data.get("items", [])
    scored: List[Tuple[int, Dict[str, Any]]] = []
    keyset = {k for k in keywords if k}
    for item in items:
        item_keys = set(item.get("keywords", []))
        score = len(keyset.intersection(item_keys))
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in scored[:top_k]]
