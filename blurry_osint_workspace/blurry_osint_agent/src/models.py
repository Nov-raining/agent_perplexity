from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class VLMPerception:
    blur_level: str
    subject_type: str
    features: List[str]
    region_hint: str
    scene_hint: str
    recognizability: float


@dataclass
class EnhancementPlan:
    steps: List[str]


@dataclass
class SearchPlan:
    engines: List[str]
    keywords: List[str]
    preprocess: List[str]


@dataclass
class SearchResult:
    engine: str
    title: str
    url: str
    snippet: str
    score: float


@dataclass
class SourceInfo:
    original_source: str
    repost_source: str
    source_confidence: str


@dataclass
class OsintMetadata:
    gps: str
    published_at: str
    platform: str
    related_text: str
    exif: str
    source_url: str
    source_info: SourceInfo
    called_apis: List[str]


@dataclass
class FusionConclusion:
    conclusion: str
    confidence: float
    evidence: List[str]


@dataclass
class IterationReport:
    iteration: int
    tools_called: List[str]
    perception: VLMPerception
    plan: SearchPlan
    osint: OsintMetadata
    conclusion: FusionConclusion
    failure_reason: str = ""
    optimization: str = ""
    second_round_result: str = ""


@dataclass
class AgentOutput:
    reports: List[IterationReport] = field(default_factory=list)
