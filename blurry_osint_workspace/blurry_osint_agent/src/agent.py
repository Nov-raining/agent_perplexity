from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import List

from .cache import (
    add_cache_entry,
    build_conclusion_from_cache,
    build_osint_from_cache,
    compute_ahash,
    find_similar,
)
from .config import (
    RECOGNIZABILITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
    TOP_K_RESULTS,
    ENGINE_PRIORITY,
    CACHE_ENABLED,
    CACHE_SIMILARITY_THRESHOLD,
)
from .models import AgentOutput, IterationReport, SearchPlan
from .tools import ToolBundle, build_mock_tools, build_real_tools


class BlurryOsintAgent:
    def __init__(self, tools: ToolBundle) -> None:
        self.tools = tools

    def run(self, image_path: str) -> AgentOutput:
        output = AgentOutput()
        perception = self.tools.vlm.perceive(image_path)

        last_confidence = 0.0
        plan = self._build_plan(perception)

        for iteration in range(1, MAX_ITERATIONS + 1):
            enhanced_path = self.tools.enhancer.apply(image_path, plan.preprocess)

            tools_called = ["Tool1-VLM"]
            if perception.recognizability < RECOGNIZABILITY_THRESHOLD:
                tools_called.append("Tool2-Enhance")
            else:
                tools_called.append("Tool2-Enhance-Optional")

            cache_hit = False
            cache_similarity = None
            if CACHE_ENABLED:
                hash_val = compute_ahash(enhanced_path)
                if hash_val is not None:
                    cached_item, dist = find_similar(hash_val, CACHE_SIMILARITY_THRESHOLD)
                    if cached_item is not None:
                        cache_hit = True
                        cache_similarity = dist
                        tools_called.append("Cache")
                        metadata = build_osint_from_cache(cached_item)
                        conclusion = build_conclusion_from_cache(cached_item)
                        results = []
                        last_confidence = conclusion.confidence
                    else:
                        results = None
                else:
                    results = None
            else:
                results = None

            if not cache_hit:
                tools_called.append("Tool3-Search")
                tools_called.append("Tool4-OSINT")
                tools_called.append("Tool5-Fusion")

                results = self.tools.searcher.search(plan.engines, plan.keywords, TOP_K_RESULTS)
                metadata = self.tools.osint.extract(results, image_path, enhanced_path)
                conclusion = self.tools.fuser.fuse(perception, results, metadata)
                last_confidence = conclusion.confidence

            failure_reason = ""
            optimization = ""
            second_round_result = ""
            if last_confidence < CONFIDENCE_THRESHOLD and iteration < MAX_ITERATIONS:
                failure_reason = self.tools.reflector.optimize(perception, last_confidence)
                optimization = self._build_optimization(plan)
                second_round_result = "已启动第二轮检索"

            report = IterationReport(
                iteration=iteration,
                tools_called=tools_called,
                perception=perception,
                plan=plan,
                osint=metadata,
                conclusion=conclusion,
                enhanced_path=enhanced_path,
                cache_hit=cache_hit,
                cache_similarity=cache_similarity if cache_similarity is not None else -1,
                failure_reason=failure_reason,
                optimization=optimization,
                second_round_result=second_round_result,
            )
            output.reports.append(report)

            if CACHE_ENABLED and not cache_hit:
                hash_val = compute_ahash(enhanced_path)
                if hash_val is not None:
                    add_cache_entry(hash_val, image_path, enhanced_path, metadata, conclusion)

            if last_confidence >= CONFIDENCE_THRESHOLD:
                break

            if iteration < MAX_ITERATIONS:
                plan = self._iterate_plan(plan, perception)
            else:
                break

        return output

    def _build_plan(self, perception) -> SearchPlan:
        engines = ENGINE_PRIORITY.get(perception.subject_type, ["Google Lens", "TinEye"])
        keywords = [perception.subject_type, perception.region_hint] + perception.features[:2]
        preprocess = self.tools.enhancer.plan(perception.blur_level, perception.recognizability).steps
        return SearchPlan(engines=engines, keywords=keywords, preprocess=preprocess)

    def _iterate_plan(self, plan: SearchPlan, perception) -> SearchPlan:
        engines = list(plan.engines)
        if "TinEye" not in engines:
            engines.append("TinEye")
        if "Lenso.ai" not in engines and perception.subject_type in {"人脸", "人像"}:
            engines.append("Lenso.ai")
        keywords = plan.keywords + ["街景", "地标" if perception.subject_type in {"建筑", "地标"} else "细节"]
        preprocess = plan.preprocess + ["局部抠图"]
        return SearchPlan(engines=engines, keywords=keywords, preprocess=preprocess)

    def _build_optimization(self, plan: SearchPlan) -> str:
        return "更换关键词、扩展引擎组合、加强局部增强"


def format_report(output: AgentOutput) -> str:
    blocks: List[str] = []
    for report in output.reports:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        blocks.append(
            "【Agent执行日志】\n"
            f"迭代轮次：第{report.iteration}轮 | 执行时间：{timestamp} | 调用工具：{', '.join(report.tools_called)}\n\n"
            "【图像感知报告】\n"
            f"模糊等级：{report.perception.blur_level} | 主体类型：{report.perception.subject_type} | 核心特征："
            f"[{', '.join(report.perception.features[:3])}] | 地域推测：{report.perception.region_hint} | "
            f"基础可识别度：{report.perception.recognizability:.1f}\n\n"
            "【首轮/迭代策略】\n"
            f"推荐引擎组合：{', '.join(report.plan.engines)} | 搜索关键词：{' '.join(report.plan.keywords)} | "
            f"图像预处理方案：{', '.join(report.plan.preprocess)}\n\n"
            "【OSINT元数据提取结果】\n"
            f"GPS坐标：{report.osint.gps} | 发布时间：{report.osint.published_at} | 来源平台：{report.osint.platform} | "
            f"关联文本：{report.osint.related_text} | EXIF信息：{report.osint.exif}\n"
            f"原始来源：{report.osint.source_info.original_source} | 转发来源：{report.osint.source_info.repost_source or '无'} | "
            f"来源可信度：{report.osint.source_info.source_confidence}\n"
            f"调用API：{', '.join(report.osint.called_apis) if report.osint.called_apis else '无'}\n"
        )
        if report.cache_hit:
            blocks.append(
                f"缓存命中：是 | 相似度距离：{report.cache_similarity}\n\n"
            )
        if report.osint.api_errors:
            error_text = " ; ".join(
                f"{e.api}:{e.category}({e.retry_count})" for e in report.osint.api_errors
            )
            blocks.append(f"API错误：{error_text}\n\n")
        else:
            blocks.append("\n")
        blocks.append(
            "【溯源结论】\n"
            f"最终结论：{report.conclusion.conclusion} | 可信度评分：{report.conclusion.confidence:.1f} | 证据链（逐条列明）："
            f"1.{report.conclusion.evidence[0]} 2.{report.conclusion.evidence[1]} 3.{report.conclusion.evidence[2]}\n"
        )
        if report.iteration == 1 and report.conclusion.confidence < CONFIDENCE_THRESHOLD:
            blocks.append(
                "\n【迭代说明（首轮失败时必填）】\n"
                f"首轮失败原因：{report.failure_reason} | 优化策略：{report.optimization} | 第二轮执行结果：{report.second_round_result}\n"
            )
    blocks.append(
        "\n【合规提示】\n"
        "图片来源仅用于溯源，不用于商业用途；图片版权归原始发布者所有，溯源结果仅供参考。"
    )
    return "\n".join(blocks).strip()


def format_report_json(output: AgentOutput) -> str:
    import json

    return json.dumps(asdict(output), ensure_ascii=False, indent=2)


def build_agent(mode: str) -> "BlurryOsintAgent":
    if mode == "real":
        return BlurryOsintAgent(build_real_tools())
    return BlurryOsintAgent(build_mock_tools())
