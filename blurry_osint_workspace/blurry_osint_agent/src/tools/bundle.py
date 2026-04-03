from __future__ import annotations

from dataclasses import dataclass

from .enhance import ImageEnhanceTool
from .fusion import FusionTool
from .osint import BaseOsintTool, OsintToolMock, OsintToolReal
from .reflect import ReflectTool
from .search import SearchTool
from .vlm import BaseVLMTool, VLMToolLMDeploy, VLMToolMock


@dataclass
class ToolBundle:
    vlm: BaseVLMTool
    enhancer: ImageEnhanceTool
    searcher: SearchTool
    osint: BaseOsintTool
    fuser: FusionTool
    reflector: ReflectTool


def build_mock_tools() -> ToolBundle:
    return ToolBundle(
        vlm=VLMToolMock(),
        enhancer=ImageEnhanceTool(),
        searcher=SearchTool(),
        osint=OsintToolMock(),
        fuser=FusionTool(),
        reflector=ReflectTool(),
    )


def build_real_tools() -> ToolBundle:
    return ToolBundle(
        vlm=VLMToolLMDeploy(),
        enhancer=ImageEnhanceTool(),
        searcher=SearchTool(),
        osint=OsintToolReal(),
        fuser=FusionTool(),
        reflector=ReflectTool(),
    )
