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


def run_demo(image, mode: str, output: str, use_langchain: bool) -> str:
    image_path = _resolve_image_path(image)
    if not image_path or not os.path.exists(image_path):
        return "未检测到有效图片路径，请重新上传。"
    if use_langchain:
        try:
            return run_with_langchain(image_path, mode, output)
        except Exception as exc:
            return f"LangChain 调用失败：{exc}"
    agent = build_agent(mode)
    result = agent.run(image_path)
    if output == "json":
        return format_report_json(result)
    return format_report(result)


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
        result = gr.Textbox(lines=20, label="输出结果")

        run_btn.click(
            fn=run_demo,
            inputs=[image, mode, output, use_langchain],
            outputs=[result],
        )

    demo.launch()


if __name__ == "__main__":
    launch()
