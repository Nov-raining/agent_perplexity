from __future__ import annotations

from typing import List

from ..models import SearchResult
from .utils import score_from_seed, seed_from_text


class SearchTool:
    def search(self, engines: List[str], keywords: List[str], top_k: int) -> List[SearchResult]:
        results: List[SearchResult] = []
        base_seed = seed_from_text("|".join(engines) + "|" + " ".join(keywords))
        for idx, engine in enumerate(engines):
            for i in range(top_k // max(1, len(engines))):
                score = score_from_seed(base_seed + idx * 31 + i * 7)
                title = f"{engine} 命中 {keywords[0]} {i}"
                url = f"https://example.com/{engine.replace(' ', '').lower()}/{base_seed % 1000}/{i}"
                snippet = f"包含特征: {', '.join(keywords[:2])}"
                results.append(SearchResult(engine=engine, title=title, url=url, snippet=snippet, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
