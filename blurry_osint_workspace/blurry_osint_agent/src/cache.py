from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .models import OsintMetadata, SourceInfo, FusionConclusion


CACHE_FILENAME = "cache.json"


def _cache_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
    os.makedirs(base, exist_ok=True)
    return base


def cache_path() -> str:
    return os.path.join(_cache_dir(), CACHE_FILENAME)


def load_cache() -> Dict[str, Any]:
    path = cache_path()
    if not os.path.exists(path):
        return {"items": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"items": []}


def save_cache(data: Dict[str, Any]) -> None:
    path = cache_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def compute_ahash(image_path: str) -> Optional[int]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small >= avg).astype(int).flatten()
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_similar(hash_val: int, threshold: int) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    data = load_cache()
    best_item = None
    best_dist = None
    for item in data.get("items", []):
        item_hash = item.get("hash")
        if item_hash is None:
            continue
        dist = hamming_distance(hash_val, int(item_hash))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_item = item
    if best_item is not None and best_dist is not None and best_dist <= threshold:
        return best_item, best_dist
    return None, None


def add_cache_entry(
    hash_val: int,
    image_path: str,
    enhanced_path: str,
    osint: OsintMetadata,
    conclusion: FusionConclusion,
) -> None:
    data = load_cache()
    items = data.get("items", [])
    items.append(
        {
            "hash": hash_val,
            "image_path": image_path,
            "enhanced_path": enhanced_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "osint": asdict(osint),
            "conclusion": asdict(conclusion),
        }
    )
    data["items"] = items[-500:]
    save_cache(data)


def build_osint_from_cache(item: Dict[str, Any]) -> OsintMetadata:
    osint = item.get("osint", {})
    source_info = osint.get("source_info", {})
    return OsintMetadata(
        gps=osint.get("gps", "N/A"),
        published_at=osint.get("published_at", "N/A"),
        platform=osint.get("platform", "Cache"),
        related_text=osint.get("related_text", ""),
        exif=osint.get("exif", "N/A"),
        source_url=osint.get("source_url", ""),
        source_info=SourceInfo(
            original_source=source_info.get("original_source", ""),
            repost_source=source_info.get("repost_source", ""),
            source_confidence=source_info.get("source_confidence", "中"),
        ),
        called_apis=["Cache"],
        api_errors=[],
    )


def build_conclusion_from_cache(item: Dict[str, Any]) -> FusionConclusion:
    conclusion = item.get("conclusion", {})
    return FusionConclusion(
        conclusion=conclusion.get("conclusion", "缓存命中"),
        confidence=float(conclusion.get("confidence", 0.5)),
        evidence=conclusion.get("evidence", ["Cache命中"]),
    )
