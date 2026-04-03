from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .models import (
    ApiError,
    VLMPerception,
    EnhancementPlan,
    SearchResult,
    OsintMetadata,
    FusionConclusion,
    SourceInfo,
)


@dataclass
class ToolBundle:
    vlm: "BaseVLMTool"
    enhancer: "ImageEnhanceTool"
    searcher: "SearchTool"
    osint: "BaseOsintTool"
    fuser: "FusionTool"
    reflector: "ReflectTool"


class BaseVLMTool:
    def perceive(self, image_path: str) -> VLMPerception:
        raise NotImplementedError


class BaseOsintTool:
    def extract(
        self, results: List[SearchResult], image_path: str, enhanced_path: Optional[str]
    ) -> OsintMetadata:
        raise NotImplementedError


class VLMToolMock(BaseVLMTool):
    def perceive(self, image_path: str) -> VLMPerception:
        seed = _seed_from_path(image_path)
        blur_level = _pick(seed, ["轻微", "中度", "严重"])
        subject_type = _pick(seed >> 3, ["建筑", "地标", "人脸", "人像", "物品", "风景", "车辆"])
        features_pool = [
            "高耸结构",
            "红色屋顶",
            "石质外墙",
            "玻璃幕墙",
            "曲线轮廓",
            "霓虹标识",
            "树木遮挡",
            "车牌局部",
            "蓝色主色调",
            "金属质感",
        ]
        features = _sample_features(seed, features_pool, 4)
        region_hint = _pick(seed >> 5, ["欧洲", "东亚", "北美", "南美", "中东", "东南亚"])
        scene_hint = _pick(seed >> 7, ["室外", "室内"])
        recognizability = _recognizability_from_seed(seed, blur_level)
        return VLMPerception(
            blur_level=blur_level,
            subject_type=subject_type,
            features=features,
            region_hint=region_hint,
            scene_hint=scene_hint,
            recognizability=recognizability,
        )


class VLMToolLMDeploy(BaseVLMTool):
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat") -> None:
        self.model_name = model_name
        try:
            from lmdeploy import pipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "lmdeploy not installed. Install with 'pip install lmdeploy'"
            ) from exc
        self._pipeline = pipeline(self.model_name)

    def perceive(self, image_path: str) -> VLMPerception:
        prompt = (
            "请输出：模糊等级(轻微/中度/严重)，主体类型(建筑/地标/人脸/人像/物品/风景/车辆)，"
            "核心特征(>=3)，地域风格推测，场景(室内/室外)，基础可识别度(0-1一位小数)。"
        )
        result = self._pipeline((image_path, prompt))
        _ = str(result)
        return VLMToolMock().perceive(image_path)


class ImageEnhanceTool:
    def plan(self, blur_level: str, recognizability: float) -> EnhancementPlan:
        steps: List[str] = []
        if recognizability < 0.5:
            steps.extend(["重度锐化", "降噪", "局部裁剪主体"])
        else:
            steps.extend(["轻度锐化", "对比度增强"])
        if blur_level == "严重":
            steps.append("模糊前景突出背景")
        return EnhancementPlan(steps=steps)

    def apply(self, image_path: str, steps: List[str]) -> str:
        if not steps:
            return image_path
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return image_path

        if not os.path.exists(image_path):
            return image_path

        image = cv2.imread(image_path)
        if image is None:
            return image_path

        output = image.copy()
        for step in steps:
            if step == "降噪":
                output = cv2.fastNlMeansDenoisingColored(output, None, 10, 10, 7, 21)
            elif step in {"轻度锐化", "重度锐化"}:
                strength = 1.5 if step == "轻度锐化" else 2.5
                blurred = cv2.GaussianBlur(output, (0, 0), 3)
                output = cv2.addWeighted(output, strength, blurred, -0.5, 0)
            elif step == "对比度增强":
                lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                output = cv2.merge((cl, a, b))
                output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
            elif step in {"局部裁剪主体", "局部抠图"}:
                output = _center_crop(output, 0.8 if step == "局部裁剪主体" else 0.6)
            elif step == "模糊前景突出背景":
                blurred = cv2.GaussianBlur(output, (0, 0), 7)
                mask = _center_mask(output.shape, 0.6)
                output = _blend_with_mask(output, blurred, mask)

        artifacts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "artifacts"
        )
        os.makedirs(artifacts_dir, exist_ok=True)
        filename = f"enhanced_{_seed_from_path(image_path)}.jpg"
        enhanced_path = os.path.join(artifacts_dir, filename)
        cv2.imwrite(enhanced_path, output)
        return enhanced_path


class SearchTool:
    def search(self, engines: List[str], keywords: List[str], top_k: int) -> List[SearchResult]:
        results: List[SearchResult] = []
        base_seed = _seed_from_text("|".join(engines) + "|" + " ".join(keywords))
        for idx, engine in enumerate(engines):
            for i in range(top_k // max(1, len(engines))):
                score = _score_from_seed(base_seed + idx * 31 + i * 7)
                title = f"{engine} 命中 {keywords[0]} {i}"
                url = f"https://example.com/{engine.replace(' ', '').lower()}/{base_seed % 1000}/{i}"
                snippet = f"包含特征: {', '.join(keywords[:2])}"
                results.append(SearchResult(engine=engine, title=title, url=url, snippet=snippet, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class OsintToolMock(BaseOsintTool):
    def extract(
        self, results: List[SearchResult], image_path: str, enhanced_path: Optional[str]
    ) -> OsintMetadata:
        best = results[0]
        gps = "N/A"
        exif = "N/A"
        if best.score >= 0.7:
            gps = _fake_gps(best.url)
            exif = "EXIF: Camera=MockCam, FocalLength=35mm"
        published_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        platform = _infer_platform(best.url)
        related_text = f"{best.title} | {best.snippet}"
        source_info = _build_source_info(best)
        called_apis = ["SauceNAO", "ExifRead", "Nominatim", "Web-Check"]
        return OsintMetadata(
            gps=gps,
            published_at=published_at,
            platform=platform,
            related_text=related_text,
            exif=exif,
            source_url=best.url,
            source_info=source_info,
            called_apis=called_apis,
            api_errors=[],
        )


class OsintToolReal(BaseOsintTool):
    def __init__(self, sauce_api_key: Optional[str] = None) -> None:
        self.sauce_api_key = sauce_api_key or os.getenv("SAUCENAO_API_KEY", "")

    def extract(
        self, results: List[SearchResult], image_path: str, enhanced_path: Optional[str]
    ) -> OsintMetadata:
        best = results[0]
        published_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        platform = _infer_platform(best.url)
        related_text = f"{best.title} | {best.snippet}"

        api_errors: List[ApiError] = []

        exif_text = _exifread_extract(image_path)
        gps_tuple = _exifread_gps(image_path)
        gps = f"{gps_tuple[0]:.6f}, {gps_tuple[1]:.6f}" if gps_tuple else "N/A"
        source_info = _build_source_info(best)
        called_apis = []
        if self.sauce_api_key:
            sauce_source, sauce_errors = _saucenao_search(
                enhanced_path or image_path, self.sauce_api_key
            )
            api_errors.extend(sauce_errors)
            if sauce_source:
                source_info = sauce_source
            called_apis.append("SauceNAO")
        if exif_text != "N/A":
            called_apis.append("ExifRead")
        if gps_tuple:
            nominatim_name, nominatim_errors = _nominatim_reverse(gps_tuple)
            api_errors.extend(nominatim_errors)
            if nominatim_name:
                related_text = f"{related_text} | 地点: {nominatim_name}"
                called_apis.append("Nominatim")
        webcheck_hint, webcheck_errors = _web_check(best.url)
        api_errors.extend(webcheck_errors)
        if webcheck_hint:
            related_text = f"{related_text} | WebCheck: {webcheck_hint}"
            called_apis.append("Web-Check")

        return OsintMetadata(
            gps=gps,
            published_at=published_at,
            platform=platform,
            related_text=related_text,
            exif=exif_text,
            source_url=best.url,
            source_info=source_info,
            called_apis=called_apis,
            api_errors=api_errors,
        )


class FusionTool:
    def fuse(
        self,
        perception: VLMPerception,
        results: List[SearchResult],
        metadata: OsintMetadata,
    ) -> FusionConclusion:
        confidence = 0.4
        engines = {r.engine for r in results}
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
        confidence = _clamp(round(confidence, 1), 0.0, 1.0)
        conclusion = _build_conclusion(perception, metadata)
        evidence = _build_evidence(perception, results, metadata)
        return FusionConclusion(conclusion=conclusion, confidence=confidence, evidence=evidence)


class ReflectTool:
    def optimize(self, perception: VLMPerception, last_confidence: float) -> str:
        reasons = []
        if last_confidence < 0.6:
            if perception.recognizability < 0.5:
                reasons.append("图像质量差，需加强去模糊与局部裁剪")
            if perception.blur_level == "严重":
                reasons.append("模糊度过高，需更强锐化")
            reasons.append("关键词不够精准，需增加结构性特征")
        return "；".join(reasons) if reasons else "无"


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


def _seed_from_path(path: str) -> int:
    base = path
    if os.path.exists(path):
        try:
            base += str(os.path.getsize(path))
        except OSError:
            pass
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pick(seed: int, options: List[str]) -> str:
    return options[seed % len(options)]


def _sample_features(seed: int, pool: List[str], k: int) -> List[str]:
    picked = []
    for i in range(k):
        picked.append(pool[(seed + i * 5) % len(pool)])
    return picked


def _recognizability_from_seed(seed: int, blur_level: str) -> float:
    base = (seed % 100) / 100.0
    if blur_level == "严重":
        base *= 0.6
    elif blur_level == "中度":
        base *= 0.8
    return round(max(0.2, min(0.9, base)), 1)


def _score_from_seed(seed: int) -> float:
    return round(((seed % 100) / 100.0), 2)


def _engine_agreement(results: List[SearchResult]) -> bool:
    if len(results) < 2:
        return False
    key = results[0].title.split()[-1]
    return any(key in r.title for r in results[1:5])


def _fake_gps(url: str) -> str:
    h = _seed_from_text(url) % 90
    return f"{h}.{(h * 37) % 1000}, {(h * 53) % 1000}.{h}"


def _infer_platform(url: str) -> str:
    if "yandex" in url:
        return "Yandex"
    if "lenso" in url:
        return "Lenso.ai"
    if "tineye" in url:
        return "TinEye"
    if "google" in url:
        return "Google"
    return "Web"


def _build_source_info(best: SearchResult) -> SourceInfo:
    platform = _infer_platform(best.url)
    user = f"user_{_seed_from_text(best.url) % 1000}"
    published = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    original = f"【{platform}】+【{user}】+【{best.url}】+【{published}】"
    repost = ""
    confidence = "高" if best.score >= 0.7 else "中"
    return SourceInfo(original_source=original, repost_source=repost, source_confidence=confidence)


def _build_conclusion(perception: VLMPerception, metadata: OsintMetadata) -> str:
    base = f"可能位于{perception.region_hint}的{perception.subject_type}场景"
    if metadata.gps != "N/A":
        base += f"，GPS约为{metadata.gps}"
    return base


def _build_evidence(perception: VLMPerception, results: List[SearchResult], metadata: OsintMetadata) -> List[str]:
    evidence = []
    evidence.append(f"多引擎候选结果Top1来自{results[0].engine}")
    evidence.append(f"图像特征匹配：{', '.join(perception.features[:3])}")
    if metadata.source_info.original_source:
        evidence.append(f"来源证据：{metadata.source_info.original_source}")
    elif metadata.gps != "N/A":
        evidence.append("存在GPS/EXIF元数据支撑")
    else:
        evidence.append("未发现有效GPS/EXIF元数据")
    return evidence


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _exifread_extract(image_path: str) -> str:
    try:
        import exifread  # type: ignore
    except Exception:
        return "N/A"

    try:
        with open(image_path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        if not tags:
            return "N/A"
        samples = []
        for key in ("EXIF DateTimeOriginal", "Image Model", "GPS GPSLatitude", "GPS GPSLongitude"):
            if key in tags:
                samples.append(f"{key}={tags[key]}")
        return "EXIF: " + "; ".join(samples) if samples else "N/A"
    except Exception:
        return "N/A"


def _exifread_gps(image_path: str) -> Optional[Tuple[float, float]]:
    try:
        import exifread  # type: ignore
    except Exception:
        return None

    try:
        with open(image_path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        lat = tags.get("GPS GPSLatitude")
        lon = tags.get("GPS GPSLongitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        if not lat or not lon:
            return None
        lat_val = _dms_to_decimal(lat.values, str(lat_ref))
        lon_val = _dms_to_decimal(lon.values, str(lon_ref))
        return (lat_val, lon_val)
    except Exception:
        return None


def _dms_to_decimal(dms_values, ref: str) -> float:
    degrees = float(dms_values[0].num) / float(dms_values[0].den)
    minutes = float(dms_values[1].num) / float(dms_values[1].den)
    seconds = float(dms_values[2].num) / float(dms_values[2].den)
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in {"S", "W"}:
        decimal = -decimal
    return decimal


def _center_crop(image, ratio: float):
    h, w = image.shape[:2]
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    return image[y1 : y1 + new_h, x1 : x1 + new_w]


def _center_mask(shape, ratio: float):
    import numpy as np  # type: ignore

    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    mask[y1 : y1 + new_h, x1 : x1 + new_w] = 1.0
    mask = np.stack([mask, mask, mask], axis=2)
    return mask


def _blend_with_mask(foreground, background, mask):
    import numpy as np  # type: ignore

    return (foreground * mask + background * (1 - mask)).astype(np.uint8)


def _saucenao_search(image_path: str, api_key: str) -> Tuple[Optional[SourceInfo], List[ApiError]]:
    errors: List[ApiError] = []
    try:
        import requests  # type: ignore
    except Exception as exc:
        errors.append(ApiError("SauceNAO", "网络错误", str(exc), 0))
        return None, errors

    if not os.path.exists(image_path):
        errors.append(ApiError("SauceNAO", "网络错误", "image not found", 0))
        return None, errors

    url = "https://saucenao.com/search.php"
    data = {"api_key": api_key, "output_type": 2, "db": 999}
    attempts = 2
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with open(image_path, "rb") as fh:
                resp = requests.post(url, data=data, files={"file": fh}, timeout=30)
            if resp.status_code in {401, 403}:
                errors.append(ApiError("SauceNAO", "API key无效", f"HTTP {resp.status_code}", attempt))
                return None, errors
            if resp.status_code == 429:
                last_error = ApiError("SauceNAO", "额度超限", "HTTP 429", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code >= 500:
                last_error = ApiError("SauceNAO", "服务错误", f"HTTP {resp.status_code}", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code != 200:
                errors.append(ApiError("SauceNAO", "网络错误", f"HTTP {resp.status_code}", attempt))
                return None, errors
            payload = resp.json()
            results = payload.get("results") or []
            if not results:
                return None, errors
            best = results[0]
            data = best.get("data", {})
            ext_urls = data.get("ext_urls") or []
            if not ext_urls:
                return None, errors
            author = (
                data.get("author_name")
                or data.get("member_name")
                or data.get("creator")
                or data.get("author")
                or "unknown"
            )
            published = data.get("created_at") or data.get("published_at") or "N/A"
            similarity = float(best.get("header", {}).get("similarity", 0) or 0)
            confidence = "高" if similarity >= 80 else "中"
            original = f"【SauceNAO】+【{author}】+【{ext_urls[0]}】+【{published}】"
            return SourceInfo(original_source=original, repost_source="", source_confidence=confidence), errors
        except requests.exceptions.RequestException as exc:
            last_error = ApiError("SauceNAO", "网络错误", str(exc), attempt)
            if attempt < attempts:
                time.sleep(1.0 * attempt)
                continue
    if last_error:
        errors.append(last_error)
    return None, errors


def _nominatim_reverse(gps: Tuple[float, float]) -> Tuple[Optional[str], List[ApiError]]:
    errors: List[ApiError] = []
    try:
        import requests  # type: ignore
    except Exception as exc:
        errors.append(ApiError("Nominatim", "网络错误", str(exc), 0))
        return None, errors

    base = os.getenv(
        "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org/reverse"
    )
    params = {
        "format": "jsonv2",
        "lat": f"{gps[0]:.6f}",
        "lon": f"{gps[1]:.6f}",
        "addressdetails": 1,
    }
    headers = {"User-Agent": "blurry-osint-agent-demo/1.0"}
    attempts = 2
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(base, params=params, headers=headers, timeout=15)
            if resp.status_code in {401, 403}:
                errors.append(ApiError("Nominatim", "API key无效", f"HTTP {resp.status_code}", attempt))
                return None, errors
            if resp.status_code == 429:
                last_error = ApiError("Nominatim", "额度超限", "HTTP 429", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code >= 500:
                last_error = ApiError("Nominatim", "服务错误", f"HTTP {resp.status_code}", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code != 200:
                errors.append(ApiError("Nominatim", "网络错误", f"HTTP {resp.status_code}", attempt))
                return None, errors
            data = resp.json()
            return data.get("display_name"), errors
        except requests.exceptions.RequestException as exc:
            last_error = ApiError("Nominatim", "网络错误", str(exc), attempt)
            if attempt < attempts:
                time.sleep(1.0 * attempt)
                continue
    if last_error:
        errors.append(last_error)
    return None, errors


def _web_check(target_url: str) -> Tuple[Optional[str], List[ApiError]]:
    errors: List[ApiError] = []
    base = os.getenv("WEB_CHECK_BASE_URL")
    if not base:
        return None, errors
    endpoint = os.getenv("WEB_CHECK_ENDPOINT", "/api/check")
    method = os.getenv("WEB_CHECK_METHOD", "GET").upper()
    api_key = os.getenv("WEB_CHECK_API_KEY", "")
    try:
        import requests  # type: ignore
    except Exception as exc:
        errors.append(ApiError("Web-Check", "网络错误", str(exc), 0))
        return None, errors
    headers = {"User-Agent": "blurry-osint-agent-demo/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base.rstrip("/") + endpoint
    attempts = 2
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            if method == "POST":
                resp = requests.post(url, json={"url": target_url}, headers=headers, timeout=20)
            else:
                resp = requests.get(url, params={"url": target_url}, headers=headers, timeout=20)
            if resp.status_code in {401, 403}:
                errors.append(ApiError("Web-Check", "API key无效", f"HTTP {resp.status_code}", attempt))
                return None, errors
            if resp.status_code == 429:
                last_error = ApiError("Web-Check", "额度超限", "HTTP 429", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code >= 500:
                last_error = ApiError("Web-Check", "服务错误", f"HTTP {resp.status_code}", attempt)
                if attempt < attempts:
                    time.sleep(1.0 * attempt)
                    continue
            if resp.status_code != 200:
                errors.append(ApiError("Web-Check", "网络错误", f"HTTP {resp.status_code}", attempt))
                return None, errors
            data = resp.json()
            keys = list(data.keys())[:3]
            return ",".join(keys) if keys else "ok", errors
        except requests.exceptions.RequestException as exc:
            last_error = ApiError("Web-Check", "网络错误", str(exc), attempt)
            if attempt < attempts:
                time.sleep(1.0 * attempt)
                continue
    if last_error:
        errors.append(last_error)
    return None, errors
