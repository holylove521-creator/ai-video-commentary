"""
AI 智能视频解说生成系统 - 主入口

用法::

    python main.py --input video.mp4 --output result.mp4 --style game
    python main.py --input video.mp4 --style sports --ref-audio voice.wav
    python main.py --input video.mp4 --fps 2.0 --no-subtitle --style vlog
"""

import argparse
import asyncio
import time
from pathlib import Path

import yaml
from loguru import logger


# ------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------

def load_config(config_path: str = "config/model_config.yaml") -> dict:
    """从 YAML 文件加载全局配置。

    Args:
        config_path: 配置文件路径，默认 ``config/model_config.yaml``。

    Returns:
        配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"[main] 配置已加载: {config_path}")
    return cfg


# ------------------------------------------------------------------
# 流水线执行
# ------------------------------------------------------------------

async def run_pipeline(args: argparse.Namespace, config: dict) -> None:
    """执行完整的四阶段视频解说生成流水线。

    Args:
        args:   命令行参数解析结果。
        config: 全局配置字典。
    """
    from utils.llm_client import create_clients, create_fast_client
    from utils.vram_manager import VRAMManager
    from pipeline.stage1_understanding import VideoUnderstanding
    from pipeline.stage2_scriptgen import ScriptGenerator
    from pipeline.stage3_tts import TTSEngine
    from pipeline.stage4_editing import VideoEditor

    total_start = time.time()
    vram = VRAMManager()
    vram.log_status()

    # 创建 LLM 客户端
    vl_client, script_client = create_clients(config)
    fast_vl_client = create_fast_client(config)

    try:
        # ----------------------------------------------------------
        # Stage 1: 视频理解
        # ----------------------------------------------------------
        logger.info("=" * 60)
        logger.info("Stage 1/4 ▶ 视频理解与场景分析")
        t1 = time.time()
        stage1 = VideoUnderstanding(vl_client, config, fast_vl_client=fast_vl_client)
        scenes = await stage1.analyze_video(
            args.input, fps_sample=args.fps
        )
        logger.info(f"Stage 1 完成，识别 {len(scenes)} 个场景  ({time.time()-t1:.1f}s)")
        vram.log_status()

        # ----------------------------------------------------------
        # Stage 2: 脚本生成
        # ----------------------------------------------------------
        logger.info("=" * 60)
        logger.info(f"Stage 2/4 ▶ 解说脚本生成（风格: {args.style}）")
        t2 = time.time()
        stage2 = ScriptGenerator(script_client, style=args.style, config=config)
        script = await stage2.generate(scenes)
        logger.info(f"Stage 2 完成，生成 {len(script)} 段脚本  ({time.time()-t2:.1f}s)")
        vram.log_status()

    finally:
        await vl_client.close()
        await script_client.close()
        if fast_vl_client is not None and fast_vl_client is not vl_client:
            await fast_vl_client.close()
        vram.force_gc()

    # ----------------------------------------------------------
    # Stage 3: 语音合成
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3/4 ▶ 语音合成（TTS）")
    t3 = time.time()
    stage3 = TTSEngine(config, ref_audio=args.ref_audio)
    script_with_audio = stage3.synthesize_all(script)
    logger.info(f"Stage 3 完成  ({time.time()-t3:.1f}s)")
    vram.log_status()

    # ----------------------------------------------------------
    # Stage 4: 剪辑合成
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 4/4 ▶ 智能剪辑与视频合成")
    t4 = time.time()
    stage4 = VideoEditor(config)

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    result_path = stage4.compose(
        video_path=args.input,
        script_with_audio=script_with_audio,
        output_path=args.output,
        no_subtitle=args.no_subtitle,
    )
    logger.info(f"Stage 4 完成  ({time.time()-t4:.1f}s)")

    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    total_elapsed = time.time() - total_start
    logger.success("=" * 60)
    logger.success(f"✅ 视频解说生成完成！")
    logger.success(f"   输入:  {args.input}")
    logger.success(f"   输出:  {result_path}")
    logger.success(f"   风格:  {args.style}")
    logger.success(f"   总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.success("=" * 60)


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

def main() -> None:
    """命令行主入口，解析参数并启动流水线。"""
    parser = argparse.ArgumentParser(
        description="AI 智能视频解说生成系统（基于 llama.cpp）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --input game.mp4 --output result.mp4 --style game
  python main.py --input vlog.mp4 --style vlog --ref-audio myvoice.wav
  python main.py --input sports.mp4 --fps 2.0 --style sports --no-subtitle
        """,
    )
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", default="output.mp4", help="输出视频路径（默认 output.mp4）")
    parser.add_argument(
        "--style",
        default="game",
        choices=["game", "sports", "vlog", "doc", "comedy"],
        help="解说风格（默认 game）",
    )
    parser.add_argument(
        "--ref-audio", default=None,
        help="声音克隆参考音频路径（3-10 秒干净人声 WAV，可选）",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="视频抽帧频率，帧/秒（默认 1.0）",
    )
    parser.add_argument(
        "--no-subtitle", action="store_true",
        help="跳过字幕生成与烧录",
    )
    parser.add_argument(
        "--config", default="config/model_config.yaml",
        help="配置文件路径（默认 config/model_config.yaml）",
    )

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"输入文件不存在: {args.input}")
        raise SystemExit(1)

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    asyncio.run(run_pipeline(args, config))


if __name__ == "__main__":
    main()
