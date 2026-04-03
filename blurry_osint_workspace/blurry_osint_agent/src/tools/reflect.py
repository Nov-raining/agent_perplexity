from __future__ import annotations


class ReflectTool:
    def optimize(self, perception, last_confidence: float) -> str:
        reasons = []
        if last_confidence < 0.6:
            if perception.recognizability < 0.5:
                reasons.append("图像质量差，需加强去模糊与局部裁剪")
            if perception.blur_level == "严重":
                reasons.append("模糊度过高，需更强锐化")
            reasons.append("关键词不够精准，需增加结构性特征")
        return "；".join(reasons) if reasons else "无"
