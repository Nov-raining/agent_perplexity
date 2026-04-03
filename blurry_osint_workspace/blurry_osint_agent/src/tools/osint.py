from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from ..models import ApiError, OsintMetadata, SearchResult, SourceInfo
from .utils import seed_from_text


class BaseOsintTool:
    def extract(
        self, results: List[SearchResult], image_path: str, enhanced_path: Optional[str]
    ) -> OsintMetadata:
        raise NotImplementedError


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


def _fake_gps(url: str) -> str:
    h = seed_from_text(url) % 90
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
    user = f"user_{seed_from_text(best.url) % 1000}"
    published = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    original = f"【{platform}】+【{user}】+【{best.url}】+【{published}】"
    repost = ""
    confidence = "高" if best.score >= 0.7 else "中"
    return SourceInfo(original_source=original, repost_source=repost, source_confidence=confidence)


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


def _saucenao_search(image_path: str, api_key: str):
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


def _nominatim_reverse(gps: Tuple[float, float]):
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


def _web_check(target_url: str):
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
