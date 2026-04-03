from __future__ import annotations

import os
from typing import Optional

import gradio as gr

from .agent import build_agent, format_report, format_report_json
from .langchain_adapter import run_with_langchain


def _resolve_image_path(image) -> Optional[str]:
    if image is None:
        return None
    if isinstance(image, str):
        return image
    if hasattr(image, "name"):
        return image.name
    return None


def run_demo(image, mode: str, output: str, use_langchain: bool):
    image_path = _resolve_image_path(image)
    if not image_path or not os.path.exists(image_path):
        return "未检测到有效图片路径，请重新上传。", None
    if use_langchain:
        try:
            result_text = run_with_langchain(image_path, mode, output)
            enhanced_path = None
        except Exception as exc:
            return f"LangChain 调用失败：{exc}", None
        return result_text, enhanced_path
    agent = build_agent(mode)
    result = agent.run(image_path)
    enhanced_path = result.reports[0].enhanced_path if result.reports else None
    if output == "json":
        return format_report_json(result), enhanced_path
    return format_report(result), enhanced_path


def launch() -> None:
    with gr.Blocks(title="Blurry OSINT Tracing Agent") as demo:
        gr.Markdown("# Blurry OSINT Tracing Agent Demo")
        gr.Markdown(
            "上传模糊图片，选择模式与输出格式，系统将执行完整闭环溯源并输出结构化结果。"
        )
        with gr.Row():
            image = gr.Image(type="filepath", label="上传图片")
        with gr.Row():
            mode = gr.Dropdown(["mock", "real"], value="mock", label="模式")
            output = gr.Dropdown(["text", "json"], value="text", label="输出格式")
            use_langchain = gr.Checkbox(value=True, label="使用 LangChain 调度")
        run_btn = gr.Button("开始溯源")
        with gr.Row():
            result = gr.Textbox(lines=20, label="输出结果")
        with gr.Row():
            enhanced = gr.Image(type="filepath", label="增强后图像")

        run_btn.click(
            fn=run_demo,
            inputs=[image, mode, output, use_langchain],
            outputs=[result, enhanced],
        )

    demo.launch()


if __name__ == "__main__":
    launch()
