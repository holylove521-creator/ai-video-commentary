"""
Web UI - Gradio 界面

提供一个全中文的 Gradio Web 界面，允许用户上传视频、
选择解说风格、上传参考音频，实时查看处理日志并下载成品视频。

启动方式::

    python web_ui/app.py
    # 浏览器访问 http://localhost:7860
"""

import asyncio
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

# 将项目根目录加入 sys.path（从 web_ui/ 子目录运行时需要）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ------------------------------------------------------------------
# 日志缓冲（供 UI 实时展示）
# ------------------------------------------------------------------

_LOG_BUFFER: list[str] = []
_MAX_LOG_LINES = 200


def _log_sink(message) -> None:
    """loguru sink，将日志追加到全局缓冲区。"""
    _LOG_BUFFER.append(message.strip())
    if len(_LOG_BUFFER) > _MAX_LOG_LINES:
        del _LOG_BUFFER[: len(_LOG_BUFFER) - _MAX_LOG_LINES]


logger.add(_log_sink, format="{time:HH:mm:ss} | {level:<7} | {message}")


def _get_log_text() -> str:
    return "\n".join(_LOG_BUFFER[-100:])


# ------------------------------------------------------------------
# 处理函数
# ------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = _ROOT / "config" / "model_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_video(
    video_file,
    style_choice: str,
    ref_audio_file,
    fps_sample: float,
    enable_subtitle: bool,
):
    """Gradio 处理函数：执行完整流水线，返回 (日志, 输出视频路径)。"""
    import gradio as gr

    if video_file is None:
        return "⚠️ 请先上传视频文件", None

    # 风格中文 → 英文 key 映射
    style_map = {
        "🎮 游戏解说": "game",
        "⚽ 体育解说": "sports",
        "📹 生活 Vlog": "vlog",
        "🎬 纪录片旁白": "doc",
        "😂 吐槽搞笑": "comedy",
    }
    style = style_map.get(style_choice, "game")
    ref_audio = ref_audio_file if ref_audio_file else None

    output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"result_{int(time.time())}.mp4")

    _LOG_BUFFER.clear()
    logger.info(f"开始处理 | 风格: {style} | FPS: {fps_sample} | 字幕: {enable_subtitle}")

    try:
        config = _load_config()

        # 动态导入（避免在 import 时报错）
        from utils.llm_client import create_clients
        from utils.vram_manager import VRAMManager
        from pipeline.stage1_understanding import VideoUnderstanding
        from pipeline.stage2_scriptgen import ScriptGenerator
        from pipeline.stage3_tts import TTSEngine
        from pipeline.stage4_editing import VideoEditor

        async def _pipeline():
            vl_client, script_client = create_clients(config)
            try:
                stage1 = VideoUnderstanding(vl_client, config)
                scenes = await stage1.analyze_video(video_file, fps_sample=fps_sample)

                stage2 = ScriptGenerator(script_client, style=style, config=config)
                script = await stage2.generate(scenes)
            finally:
                await vl_client.close()
                await script_client.close()
                VRAMManager().force_gc()

            stage3 = TTSEngine(config, ref_audio=ref_audio)
            script_with_audio = stage3.synthesize_all(script)

            stage4 = VideoEditor(config)
            return stage4.compose(
                video_path=video_file,
                script_with_audio=script_with_audio,
                output_path=output_path,
                no_subtitle=not enable_subtitle,
            )

        result = asyncio.run(_pipeline())
        logger.success(f"处理完成 → {result}")
        return _get_log_text(), result

    except Exception as exc:
        logger.error(f"处理失败: {exc}")
        return _get_log_text(), None


# ------------------------------------------------------------------
# UI 构建
# ------------------------------------------------------------------

def create_ui():
    """构建并返回 Gradio Blocks 界面。"""
    import gradio as gr

    with gr.Blocks(title="🎬 AI 智能视频解说生成系统") as demo:
        gr.Markdown(
            """
# 🎬 AI 智能视频解说生成系统

> 基于 **llama.cpp** 本地大模型，自动为视频生成解说配音与字幕
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传文件")
                video_input = gr.Video(label="上传视频", sources=["upload"])
                ref_audio_input = gr.Audio(
                    label="参考声音（可选，声音克隆用）",
                    type="filepath",
                    sources=["upload"],
                )

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 参数设置")
                style_dropdown = gr.Dropdown(
                    label="解说风格",
                    choices=[
                        "🎮 游戏解说",
                        "⚽ 体育解说",
                        "📹 生活 Vlog",
                        "🎬 纪录片旁白",
                        "😂 吐槽搞笑",
                    ],
                    value="🎮 游戏解说",
                )
                fps_slider = gr.Slider(
                    label="抽帧频率（帧/秒）",
                    minimum=0.5,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                )
                subtitle_toggle = gr.Checkbox(
                    label="生成字幕", value=True
                )
                run_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

        gr.Markdown("### 📊 处理日志")
        log_output = gr.Textbox(
            label="实时日志",
            lines=12,
            max_lines=20,
            interactive=False,
        )

        gr.Markdown("### 🎬 输出结果")
        with gr.Row():
            video_output = gr.Video(label="生成视频预览")

        # 刷新日志（每 2 秒）
        demo.load(fn=_get_log_text, outputs=log_output, every=2)

        run_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                style_dropdown,
                ref_audio_input,
                fps_slider,
                subtitle_toggle,
            ],
            outputs=[log_output, video_output],
        )

    return demo


# ------------------------------------------------------------------
# 启动入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import gradio as gr  # noqa: F401  验证已安装
    except ImportError:
        print("请先安装 gradio：pip install gradio")
        sys.exit(1)

    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
