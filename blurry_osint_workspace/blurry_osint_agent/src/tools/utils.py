from __future__ import annotations

import hashlib
import os
from typing import List


def seed_from_path(path: str) -> int:
    base = path
    if os.path.exists(path):
        try:
            base += str(os.path.getsize(path))
        except OSError:
            pass
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def seed_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def pick(seed: int, options: List[str]) -> str:
    return options[seed % len(options)]


def sample_features(seed: int, pool: List[str], k: int) -> List[str]:
    picked = []
    for i in range(k):
        picked.append(pool[(seed + i * 5) % len(pool)])
    return picked


def recognizability_from_seed(seed: int, blur_level: str) -> float:
    base = (seed % 100) / 100.0
    if blur_level == "严重":
        base *= 0.6
    elif blur_level == "中度":
        base *= 0.8
    return round(max(0.2, min(0.9, base)), 1)


def score_from_seed(seed: int) -> float:
    return round(((seed % 100) / 100.0), 2)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
