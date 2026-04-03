from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .memory import load_memory, save_memory


def _vector_path() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "rag")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "vector_store.json")


def _build_doc(item: Dict[str, Any]) -> str:
    keywords = " ".join(item.get("keywords", []))
    conclusion = item.get("conclusion", "")
    source_url = item.get("source_url", "")
    return f"{keywords} {conclusion} {source_url}".strip()


def rebuild_vector_store() -> Dict[str, Any]:
    data = load_memory()
    items = data.get("items", [])
    store = {"docs": [_build_doc(i) for i in items]}
    _save_vector_store(store)
    return store


def _load_vector_store() -> Dict[str, Any]:
    path = _vector_path()
    if not os.path.exists(path):
        return rebuild_vector_store()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return rebuild_vector_store()


def _save_vector_store(store: Dict[str, Any]) -> None:
    with open(_vector_path(), "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=2)


def add_to_vector_store(item: Dict[str, Any]) -> None:
    store = _load_vector_store()
    docs = store.get("docs", [])
    docs.append(_build_doc(item))
    store["docs"] = docs[-500:]
    _save_vector_store(store)


def retrieve_similar(text: str, top_k: int = 3) -> List[int]:
    store = _load_vector_store()
    docs = store.get("docs", [])
    if not docs:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    except Exception:
        return []

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(docs + [text])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in ranked[:top_k] if score > 0]
