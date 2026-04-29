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
    from pipeline import benchmark, media_probe, asr_stage, schema

    # 并发参数优先级：命令行 > 配置文件 > 默认
    video_cfg = config.get("video", {})
    max_concurrent_vl = args.max_concurrent_vl if args.max_concurrent_vl is not None else video_cfg.get("max_concurrent_frames", 8)
    max_concurrent_tts = args.max_concurrent_tts if args.max_concurrent_tts is not None else config.get("tts", {}).get("max_concurrent", 4)
    import psutil
    import GPUtil
    logger.info(f"[并发] VL并发: {max_concurrent_vl} | TTS并发: {max_concurrent_tts}")
    # 输出系统资源
    cpu_count = psutil.cpu_count(logical=True)
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        logger.info(f"[GPU] {gpu.name} | 显存: {gpu.memoryFree:.1f}MB / {gpu.memoryTotal:.1f}MB | 利用率: {gpu.load*100:.1f}%")
    logger.info(f"[CPU] 逻辑核数: {cpu_count} | 当前负载: {psutil.getloadavg()}")

    total_start = time.time()
    vram = VRAMManager()
    vram.log_status()

    # 创建 LLM 客户端
    vl_client, script_client = create_clients(config)
    fast_vl_client = create_fast_client(config)

    import os
    from pipeline import schema
    # 断点续跑与中间结果自动保存/加载
    try:
        # Stage 1: EventBlock
        if args.event_json and os.path.exists(args.event_json):
            logger.info(f"[Batch5] 加载 EventBlock JSON: {args.event_json}")
            with open(args.event_json, encoding="utf-8") as f:
                event_blocks = schema.schema_list_from_json(schema.EventBlock, f.read())
        else:
            logger.info("=" * 60)
            logger.info("Stage 1/4 ▶ 视频理解与场景分析")
            t1 = time.time()
            # 传递并发参数
            stage1 = VideoUnderstanding(vl_client, config, fast_vl_client=fast_vl_client)
            stage1._max_concurrent = max_concurrent_vl
            event_blocks = await stage1.analyze_video(args.input, fps_sample=args.fps)
            logger.info(f"Stage 1 完成，识别 {len(event_blocks)} 个场景  ({time.time()-t1:.1f}s)")
            # 自动保存
            out_json = f"outputs/event_blocks_{int(time.time())}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                f.write(schema.schema_list_to_json(event_blocks))
            logger.info(f"[Batch5] EventBlock 已保存: {out_json}")
        if getattr(args, "benchmark", False):
            benchmark.benchmark.mark("after_stage1")

        # Stage 2: NarrationSegment
        if args.script_json and os.path.exists(args.script_json):
            logger.info(f"[Batch5] 加载 NarrationSegment JSON: {args.script_json}")
            with open(args.script_json, encoding="utf-8") as f:
                narration_segments = schema.schema_list_from_json(schema.NarrationSegment, f.read())
        else:
            logger.info("=" * 60)
            logger.info(f"Stage 2/4 ▶ 解说脚本生成（风格: {args.style}）")
            t2 = time.time()
            stage2 = ScriptGenerator(script_client, style=args.style, config=config)
            narration_segments = await stage2.generate(event_blocks)
            logger.info(f"Stage 2 完成，生成 {len(narration_segments)} 段脚本  ({time.time()-t2:.1f}s)")
            out_json = f"outputs/narration_segments_{int(time.time())}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                f.write(schema.schema_list_to_json(narration_segments))
            logger.info(f"[Batch5] NarrationSegment 已保存: {out_json}")
        if getattr(args, "benchmark", False):
            benchmark.benchmark.mark("after_stage2")

    finally:
        await vl_client.close()
        await script_client.close()
        if fast_vl_client is not None and fast_vl_client is not vl_client:
            await fast_vl_client.close()
        vram.force_gc()

    # Stage 3: MixSegment
    if args.mix_json and os.path.exists(args.mix_json):
        logger.info(f"[Batch5] 加载 MixSegment JSON: {args.mix_json}")
        with open(args.mix_json, encoding="utf-8") as f:
            mix_segments = schema.schema_list_from_json(schema.MixSegment, f.read())
    else:
        logger.info("=" * 60)
        logger.info("Stage 3/4 ▶ 语音合成（TTS）")
        t3 = time.time()
        stage3 = TTSEngine(config, ref_audio=args.ref_audio)
            # 传递并发参数（如需多线程/多进程可在 synthesize_all 内实现）
        stage3._max_concurrent = max_concurrent_tts
        mix_segments = stage3.synthesize_all(narration_segments)
        logger.info(f"Stage 3 完成  ({time.time()-t3:.1f}s)")
        out_json = f"outputs/mix_segments_{int(time.time())}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(schema.schema_list_to_json(mix_segments))
        logger.info(f"[Batch5] MixSegment 已保存: {out_json}")
    if getattr(args, "benchmark", False):
        benchmark.benchmark.mark("after_stage3")

    # Stage 4: 剪辑合成
    logger.info("=" * 60)
    logger.info("Stage 4/4 ▶ 智能剪辑与视频合成")
    t4 = time.time()
    stage4 = VideoEditor(config)

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    result_path = stage4.compose(
        video_path=args.input,
        mix_segments=mix_segments,
        output_path=args.output,
        no_subtitle=args.no_subtitle,
    )
    logger.info(f"Stage 4 完成  ({time.time()-t4:.1f}s)")
    if getattr(args, "benchmark", False):
        benchmark.benchmark.mark("after_stage4")

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
    parser.add_argument(
        "--benchmark", action="store_true",
        help="输出各阶段耗时统计（可选）",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="仅探测输入视频基础信息并退出",
    )
    parser.add_argument(
        "--asr-audio", default=None,
        help="仅对指定音频文件做ASR转写并退出（可选）",
    )
    parser.add_argument(
        "--resume-stage", default=None, choices=[None, "event", "script", "mix"],
        help="从指定阶段/中间结果文件恢复（event/script/mix）",
    )
    parser.add_argument(
        "--event-json", default=None, help="EventBlock JSON 路径（跳过 Stage1）"
    )
    parser.add_argument(
        "--script-json", default=None, help="NarrationSegment JSON 路径（跳过 Stage1/2）"
    )
    parser.add_argument(
        "--mix-json", default=None, help="MixSegment JSON 路径（跳过 Stage1/2/3）"
    )
    parser.add_argument(
        "--max-concurrent-vl", type=int, default=None, help="VL推理最大并发数（覆盖配置文件）"
    )
    parser.add_argument(
        "--max-concurrent-tts", type=int, default=None, help="TTS合成最大并发数（覆盖配置文件）"
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

    # Batch 2: probe/media/asr 分支
    if args.probe:
        from pipeline import media_probe
        info = media_probe.probe_video(args.input)
        print("[Probe] 视频信息:", info)
        return
    if args.asr_audio:
        from pipeline import asr_stage
        asr = asr_stage.ASRStage()
        result = asr.transcribe(args.asr_audio)
        print("[ASR] 识别结果:", result)
        return

    asyncio.run(run_pipeline(args, config))


if __name__ == "__main__":
    main()
