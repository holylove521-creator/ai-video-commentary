"""
Stage 1: 视频理解与场景分析（优化版 - 两阶段流水线）

耗时目标：4K 2小时电影 ≤ 30 分钟

流程:
    Phase 0 - 场景变化检测（CPU, ~2min）
        SceneDetector 基于直方图差异，从全量帧中提取关键帧
        典型 2h 电影：7200帧 → ~300 关键帧（削减 96%）

    Phase 1 - 快速粗筛（VL-7B, 720p, 并发8, ~5min）
        轻量 Prompt，只输出 highlight_score / emotion / cut_point
        筛出 highlight_score >= threshold 的高光场景

    Phase 2 - 精准深析（VL-32B, 1080p, 并发2, ~8min）
        对高光场景生成完整的场景描述，供 Stage 2 脚本生成使用

总计：~15-25 分钟（目标 ≤ 30 分钟）
"""

import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm

from utils.frame_extractor import FrameExtractor
from utils.llm_client import LlamaCppClient
from utils.scene_detector import SceneDetector
from pipeline.schema import EventBlock


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

# Phase 1：轻量粗筛 Prompt（输出极少 token，速度优先）
FAST_ANALYSIS_PROMPT = """\
分析这张视频截图，仅返回 JSON，不要其他文字：
{
  "highlight_score": <0-10整数，10最精彩>,
  "emotion": "excited|calm|serious|funny|tense|neutral",
  "cut_point": <true|false>
}
"""

# Phase 2：完整深析 Prompt（质量优先）
SCENE_ANALYSIS_PROMPT = """\
请分析这张视频截图，以 JSON 格式返回以下字段（仅返回 JSON，不要其他文字）：

{
  "scene_desc": "场景整体描述（1-2句话）",
  "main_subject": "画面主体（人物/物体/场景）",
  "action": "正在发生的动作或事件",
  "emotion": "画面情绪氛围（excited/calm/serious/funny/tense/neutral 之一）",
  "highlight_score": <0-10的整数，10表示极度精彩>,
  "cut_point": <true/false，是否适合作为剪辑切入点>
}
"""


# ------------------------------------------------------------------
# 主类
# ------------------------------------------------------------------

class VideoUnderstanding:
    """两阶段视频理解与场景分析器。

    Args:
        vl_client:       VL-32B 精分客户端。
        config:          全局配置字典。
        fast_vl_client:  VL-7B 粗筛客户端（可选，不传则退化为单阶段 32B）。
    """

    def __init__(
        self,
        vl_client: LlamaCppClient,
        config: dict,
        fast_vl_client: Optional[LlamaCppClient] = None,
    ) -> None:
        self._client = vl_client
        self._fast_client = fast_vl_client or vl_client
        self._config = config

        video_cfg = config.get("video", {})
        scene_cfg = config.get("scene_detection", {})
        paths_cfg = config.get("paths", {})

        self._fps_sample: float = video_cfg.get("fps_sample", 1.0)
        self._frame_quality: int = video_cfg.get("frame_quality", 85)
        self._fast_resize: tuple[int, int] = tuple(video_cfg.get("fast_resize", [1280, 720]))
        self._deep_resize: tuple[int, int] = tuple(video_cfg.get("deep_resize", [1920, 1080]))
        self._max_concurrent: int = video_cfg.get("max_concurrent_frames", 8)
        self._max_concurrent_deep: int = video_cfg.get("max_concurrent_deep", 2)

        self._scene_detection_enabled: bool = scene_cfg.get("enabled", True)
        self._max_concurrent: int = video_cfg.get("max_concurrent_frames", 8)
        self._max_concurrent: int = video_cfg.get("max_concurrent", 8)
        self._min_scene_gap: float = scene_cfg.get("min_scene_gap", 2.0)
        self._top_score_threshold: int = scene_cfg.get("top_score_threshold", 7)

        temp_dir = paths_cfg.get("temp_dir", "/tmp/ai_video_tmp")
        self._temp_frames_dir = f"{temp_dir}/frames"

        self._extractor = FrameExtractor(temp_dir=self._temp_frames_dir)
        self._detector = SceneDetector(
            threshold=self._scene_threshold,
            min_scene_gap=self._min_scene_gap,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def analyze_video(
        self,
        video_path: str,
        fps_sample: Optional[float] = None,
        max_concurrent: int = None,
    ) -> list[EventBlock]:
        """端到端分析视频，返回 EventBlock 列表。"""
        if self._scene_detection_enabled:
            return await self._analyze_two_phase(video_path)
        else:
            return await self._analyze_legacy(video_path, fps_sample)

    # ------------------------------------------------------------------
    # 两阶段流水线
    # ------------------------------------------------------------------

    async def _analyze_two_phase(self, video_path: str) -> list[EventBlock]:
        """两阶段流水线：场景检测 → 粗筛 → 精分。"""
        import time
        t0 = time.time()

        # ── Phase 0: 场景变化检测（CPU）──
        logger.info("[Stage1] Phase 0: 场景变化检测...")
        keyframes = self._detector.detect_keyframes(
            video_path,
            temp_dir=self._temp_frames_dir,
            jpeg_quality=self._frame_quality,
        )
        if not keyframes:
            logger.warning("[Stage1] 未检测到关键帧，返回空场景列表")
            return []
        logger.info(f"[Stage1] Phase 0 完成: {len(keyframes)} 帧  "
                    f"耗时 {time.time()-t0:.1f}s")

        # ── Phase 1: VL-7B 快速粗筛（720p）──
        t1 = time.time()
        logger.info(f"[Stage1] Phase 1: VL-7B 粗筛 {len(keyframes)} 帧 "
                    f"(并发={self._max_concurrent})...")
        fast_results = await self._batch_analyze(
            frames=keyframes,
            client=self._fast_client,
            prompt=FAST_ANALYSIS_PROMPT,
            max_tokens=64,
            semaphore_count=self._max_concurrent,
            resize=self._fast_resize,
            desc="粗筛",
        )
        logger.info(f"[Stage1] Phase 1 完成  耗时 {time.time()-t1:.1f}s")

        # ── Phase 2: VL-32B 精分高光场景（1080p）──
        t2 = time.time()
        top_frames = [
            f for f in fast_results
            if f.get("highlight_score", 0) >= self._top_score_threshold
        ]
        logger.info(
            f"[Stage1] Phase 2: VL-32B 精分 {len(top_frames)} 帧 "
            f"(highlight≥{self._top_score_threshold}, 并发={self._max_concurrent_deep})..."
        )

        deep_map: dict[int, dict] = {}
        if top_frames:
            deep_results = await self._batch_analyze(
                frames=top_frames,
                client=self._client,
                prompt=SCENE_ANALYSIS_PROMPT,
                max_tokens=256,
                semaphore_count=self._max_concurrent_deep,
                resize=self._deep_resize,
                desc="精分",
            )
            deep_map = {f["frame_idx"]: f for f in deep_results}

        # 用精分结果覆盖粗筛结果
        merged = [deep_map.get(f["frame_idx"], f) for f in fast_results]
        merged.sort(key=lambda x: x["frame_idx"])

        logger.info(f"[Stage1] Phase 2 完成  耗时 {time.time()-t2:.1f}s")

        scenes = self._merge_scenes(merged)
        total = time.time() - t0
        logger.success(
            f"[Stage1] 全部完成  场景数={len(scenes)}  "
            f"总耗时={total/60:.1f}min"
        )
        return scenes

    # ------------------------------------------------------------------
    # 兼容旧版逐帧分析（scene_detection.enabled=false 时使用）
    # ------------------------------------------------------------------

    async def _analyze_legacy(
        self,
        video_path: str,
        fps_sample: Optional[float] = None,
    ) -> list[EventBlock]:
        fps = fps_sample if fps_sample is not None else self._fps_sample
        logger.info(f"[Stage1] 传统模式: fps_sample={fps}")
        frames = self._extractor.extract(
            video_path, fps_sample=fps, quality=self._frame_quality
        )
        if not frames:
            return []
        results = await self._batch_analyze(
            frames=frames,
            client=self._client,
            prompt=SCENE_ANALYSIS_PROMPT,
            max_tokens=256,
            semaphore_count=self._max_concurrent,
            resize=self._deep_resize,
            desc="帧分析",
        )
        results.sort(key=lambda x: x["frame_idx"])
        return self._merge_scenes(results)

    # ------------------------------------------------------------------
    # 通用批量推理
    # ------------------------------------------------------------------

    async def _batch_analyze(
        self,
        frames: list[dict],
        client: LlamaCppClient,
        prompt: str,
        max_tokens: int,
        semaphore_count: int,
        resize: tuple[int, int],
        desc: str = "分析",
    ) -> list[dict]:
        """并发分析帧列表。"""
        sem = asyncio.Semaphore(semaphore_count)

        async def _analyze_one(frame: dict) -> dict:
            async with sem:
                return await self._analyze_frame(
                    frame, client, prompt, max_tokens, resize
                )

        tasks = [_analyze_one(f) for f in frames]
        results: list[dict] = []
        for coro in async_tqdm.as_completed(tasks, desc=desc, total=len(tasks)):
            results.append(await coro)
        return results

    async def _analyze_frame(
        self,
        frame: dict,
        client: LlamaCppClient,
        prompt: str,
        max_tokens: int,
        resize: tuple[int, int],
    ) -> dict:
        """分析单帧：resize → base64 → VL 推理 → JSON 解析。"""
        try:
            # resize 帧（原始 JPEG 重新解码缩放）
            img = cv2.imread(frame["path"])
            if img is None:
                raise ValueError(f"无法读取帧图片: {frame['path']}")
            # resize 参数为 (width, height)；img.shape 为 (height, width, ch)
            if img.shape[:2] != (resize[1], resize[0]):
                img = cv2.resize(img, resize)
                # 编码为临时 JPEG bytes 发送，不写盘
                _, buf = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                img_b64 = base64.b64encode(buf.tobytes()).decode()
            else:
                # 直接读原文件
                img_b64 = base64.b64encode(
                    Path(frame["path"]).read_bytes()
                ).decode()

            raw = await client.vision_chat_b64(
                prompt=prompt,
                image_b64=img_b64,
                temperature=0.1,
                max_tokens=max_tokens,
            )

            # JSON 容错解析
            try:
                analysis = json.loads(raw)
            except json.JSONDecodeError:
                s, e = raw.find("{"), raw.rfind("}")
                if s != -1 and e > s:
                    analysis = json.loads(raw[s:e+1])
                else:
                    raise ValueError(f"无有效 JSON: {raw[:100]}")

        except Exception as exc:
            logger.warning(f"[Stage1] 帧 {frame['frame_idx']} 分析失败: {exc}")
            analysis = {
                "scene_desc": "",
                "main_subject": "",
                "action": "",
                "emotion": "neutral",
                "highlight_score": 0,
                "cut_point": False,
            }

        return {**frame, **analysis}

    # ------------------------------------------------------------------
    # 场景合并
    # ------------------------------------------------------------------

    def _merge_scenes(self, analyzed_frames: list[dict]) -> list[EventBlock]:
        """将连续帧合并为 EventBlock（时间间隔≤3s 且情绪相同）。"""
        if not analyzed_frames:
            return []

        scenes: list[EventBlock] = []
        current: list[dict] = [analyzed_frames[0]]

        for frame in analyzed_frames[1:]:
            prev = current[-1]
            gap = frame["timestamp"] - prev["timestamp"]
            if gap <= 3.0 and frame.get("emotion") == prev.get("emotion"):
                current.append(frame)
            else:
                scenes.append(self._build_scene(current))
                current = [frame]

        scenes.append(self._build_scene(current))
        return scenes

    @staticmethod
    def _build_scene(frames: list[dict]) -> EventBlock:
        scores = [f.get("highlight_score", 0) for f in frames]
        avg = sum(scores) / len(scores) if scores else 0.0
        rep = frames[len(frames) // 2]
        # 组装 EventBlock
        return EventBlock(
            start_time=frames[0]["timestamp"],
            end_time=frames[-1]["timestamp"],
            type=rep.get("action", "scene"),
            summary=rep.get("scene_desc", ""),
            characters=[rep.get("main_subject", "")] if rep.get("main_subject") else [],
            asr_transcript=None,
            visual_tags=[rep.get("emotion", "neutral")],
            extra={
                "avg_highlight": round(avg, 2),
                "has_cut_point": any(f.get("cut_point", False) for f in frames),
                "frame_count": len(frames),
                "frames": frames,
            }
        )


# ------------------------------------------------------------------
# 独立测试入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import yaml

    async def _test():
        if len(sys.argv) < 2:
            print("用法: python -m pipeline.stage1_understanding <video_path>")
            sys.exit(1)

        with open("config/model_config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        from utils.llm_client import create_clients, create_fast_client
        vl_client, _ = create_clients(cfg)
        fast_client = create_fast_client(cfg)

        try:
            stage1 = VideoUnderstanding(vl_client, cfg, fast_vl_client=fast_client)
            scenes = await stage1.analyze_video(sys.argv[1])
            for i, s in enumerate(scenes, 1):
                print(
                    f"场景{i:03d} [{s.start_time:.1f}s-{s.end_time:.1f}s]  "
                    f"亮点={s.extra.get('avg_highlight', 0)}  {s.summary[:50]}"
                )
        finally:
            await vl_client.close()
            if fast_client is not vl_client:
                await fast_client.close()

    asyncio.run(_test())
