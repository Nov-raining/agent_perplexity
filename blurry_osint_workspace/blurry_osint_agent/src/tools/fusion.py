from __future__ import annotations

from ..models import FusionConclusion
from .utils import clamp


class FusionTool:
    def fuse(self, perception, results, metadata) -> FusionConclusion:
        confidence = 0.4
        engines = {r.engine for r in results} if results else set()
        if len(engines) >= 2 and _engine_agreement(results):
            confidence += 0.3
        if metadata.gps != "N/A" or metadata.exif != "N/A":
            confidence += 0.2
        if metadata.source_info.source_confidence == "高":
            confidence += 0.1
        if len(engines) == 1 and metadata.gps == "N/A":
            confidence -= 0.2
        if perception.blur_level == "严重":
            confidence -= 0.3
        elif perception.blur_level == "中度":
            confidence -= 0.1
        if perception.recognizability < 0.5:
            confidence -= 0.1
        confidence = clamp(round(confidence, 1), 0.0, 1.0)
        conclusion = _build_conclusion(perception, metadata)
        evidence = _build_evidence(perception, results, metadata)
        return FusionConclusion(conclusion=conclusion, confidence=confidence, evidence=evidence)


def _engine_agreement(results) -> bool:
    if not results or len(results) < 2:
        return False
    key = results[0].title.split()[-1]
    return any(key in r.title for r in results[1:5])


def _build_conclusion(perception, metadata) -> str:
    base = f"可能位于{perception.region_hint}的{perception.subject_type}场景"
    if metadata.gps != "N/A":
        base += f"，GPS约为{metadata.gps}"
    return base


def _build_evidence(perception, results, metadata):
    evidence = []
    if results:
        evidence.append(f"多引擎候选结果Top1来自{results[0].engine}")
    else:
        evidence.append("缓存命中，无需外部搜索")
    evidence.append(f"图像特征匹配：{', '.join(perception.features[:3])}")
    if metadata.source_info.original_source:
        evidence.append(f"来源证据：{metadata.source_info.original_source}")
    elif metadata.gps != "N/A":
        evidence.append("存在GPS/EXIF元数据支撑")
    else:
        evidence.append("未发现有效GPS/EXIF元数据")
    return evidence
