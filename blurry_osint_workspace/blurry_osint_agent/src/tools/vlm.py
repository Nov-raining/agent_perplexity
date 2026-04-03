from __future__ import annotations

from .utils import pick, recognizability_from_seed, sample_features, seed_from_path
from ..models import VLMPerception


class BaseVLMTool:
    def perceive(self, image_path: str) -> VLMPerception:
        raise NotImplementedError


class VLMToolMock(BaseVLMTool):
    def perceive(self, image_path: str) -> VLMPerception:
        seed = seed_from_path(image_path)
        blur_level = pick(seed, ["轻微", "中度", "严重"])
        subject_type = pick(seed >> 3, ["建筑", "地标", "人脸", "人像", "物品", "风景", "车辆"])
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
        features = sample_features(seed, features_pool, 4)
        region_hint = pick(seed >> 5, ["欧洲", "东亚", "北美", "南美", "中东", "东南亚"])
        scene_hint = pick(seed >> 7, ["室外", "室内"])
        recognizability = recognizability_from_seed(seed, blur_level)
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
