from __future__ import annotations

import os
from typing import List

from ..models import EnhancementPlan
from .utils import seed_from_path


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
        filename = f"enhanced_{seed_from_path(image_path)}.jpg"
        enhanced_path = os.path.join(artifacts_dir, filename)
        cv2.imwrite(enhanced_path, output)
        return enhanced_path


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
