"""
Stage 1: 视频理解与场景分析

使用 llama.cpp 驱动的多模态视觉语言模型（Qwen2.5-VL）逐帧分析视频，
识别场景内容、主体动作、情绪氛围与高光分，最终合并为连续场景列表。

流程:
    1. FrameExtractor 按指定 fps 抽取帧图片
    2. asyncio.Semaphore 控制并发，批量调用 VL 模型
    3. 解析 JSON 响应，容错处理
    4. 将连续帧合并为场景，计算平均高光分
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm

from utils.frame_extractor import FrameExtractor
from utils.llm_client import LlamaCppClient


# ------------------------------------------------------------------
# 分析提示词
# ------------------------------------------------------------------

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


class VideoUnderstanding:
    """视频逐帧理解与场景分析器。

    Args:
        vl_client:  连接 VL llama-server 的异步客户端。
        config:     全局配置字典（从 model_config.yaml 读取）。
    """

    def __init__(self, vl_client: LlamaCppClient, config: dict) -> None:
        self._client = vl_client
        self._config = config
        video_cfg = config.get("video", {})
        self._fps_sample: float = video_cfg.get("fps_sample", 1.0)
        self._frame_quality: int = video_cfg.get("frame_quality", 85)
        self._max_concurrent: int = video_cfg.get("max_concurrent_frames", 4)
        temp_dir = config.get("paths", {}).get("temp_dir", "/tmp/ai_video_tmp")
        self._extractor = FrameExtractor(temp_dir=f"{temp_dir}/frames")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        video_path: str,
        fps_sample: Optional[float] = None,
    ) -> list[dict]:
        """抽取视频帧。

        Args:
            video_path: 视频文件路径。
            fps_sample: 覆盖配置中的抽帧频率（可选）。

        Returns:
            帧信息列表，参见 :class:`~utils.frame_extractor.FrameExtractor`。
        """
        fps = fps_sample if fps_sample is not None else self._fps_sample
        logger.info(f"[Stage1] 开始抽帧: {video_path}  fps_sample={fps}")
        return self._extractor.extract(
            video_path, fps_sample=fps, quality=self._frame_quality
        )

    async def analyze_frame(self, frame: dict) -> dict:
        """分析单帧图片，返回场景分析结果。

        Args:
            frame: 帧信息字典 ``{"frame_idx", "timestamp", "path"}``。

        Returns:
            合并了原始帧信息和 VL 分析结果的字典。
            若解析失败，返回默认值（highlight_score=0, cut_point=False）。
        """
        try:
            raw = await self._client.vision_chat(
                prompt=SCENE_ANALYSIS_PROMPT,
                image_path=frame["path"],
                temperature=0.1,
                max_tokens=256,
            )
            # 容错：先尝试直接解析，再提取第一个完整 JSON 对象（从第一个 { 到最后一个 }）
            try:
                analysis = json.loads(raw)
            except json.JSONDecodeError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    analysis = json.loads(raw[start : end + 1])
                else:
                    raise ValueError(f"未找到 JSON: {raw[:200]}")
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            logger.warning(
                f"[Stage1] 帧 {frame['frame_idx']} 分析失败: {exc}，使用默认值"
            )
            analysis = {
                "scene_desc": "",
                "main_subject": "",
                "action": "",
                "emotion": "neutral",
                "highlight_score": 0,
                "cut_point": False,
            }

        return {**frame, **analysis}

    async def analyze_video(
        self,
        video_path: str,
        fps_sample: Optional[float] = None,
    ) -> list[dict]:
        """端到端分析视频，返回场景列表。

        先抽帧，再并发分析，最后合并连续帧为场景。

        Args:
            video_path: 视频文件路径。
            fps_sample: 覆盖默认抽帧频率（可选）。

        Returns:
            场景列表，每项包含 ``start``、``end``、``avg_highlight_score`` 等字段。
        """
        frames = self.extract_frames(video_path, fps_sample)
        if not frames:
            logger.warning("[Stage1] 未抽到任何帧，返回空场景列表")
            return []

        sem = asyncio.Semaphore(self._max_concurrent)

        async def _bounded_analyze(frame: dict) -> dict:
            async with sem:
                return await self.analyze_frame(frame)

        tasks = [_bounded_analyze(f) for f in frames]
        analyzed: list[dict] = []
        for coro in async_tqdm.as_completed(tasks, desc="视频帧分析", total=len(tasks)):
            result = await coro
            analyzed.append(result)

        # 按原始帧序号排序（as_completed 不保证顺序）
        analyzed.sort(key=lambda x: x["frame_idx"])

        scenes = self._merge_scenes(analyzed)
        logger.success(f"[Stage1] 分析完成，共识别 {len(scenes)} 个场景")
        return scenes

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _merge_scenes(self, analyzed_frames: list[dict]) -> list[dict]:
        """将连续帧合并为场景，计算平均高光分。

        合并策略：相邻帧之间时间间隔 ≤ 3 秒且情绪相同则归为同一场景。

        Args:
            analyzed_frames: 已分析的帧列表（按时间升序）。

        Returns:
            场景列表，每项格式::

                {
                    "start":              float,
                    "end":                float,
                    "scene_desc":         str,
                    "main_subject":       str,
                    "action":             str,
                    "emotion":            str,
                    "avg_highlight_score": float,
                    "has_cut_point":      bool,
                    "frame_count":        int,
                }
        """
        if not analyzed_frames:
            return []

        scenes: list[dict] = []
        current_frames: list[dict] = [analyzed_frames[0]]

        for frame in analyzed_frames[1:]:
            prev = current_frames[-1]
            gap = frame["timestamp"] - prev["timestamp"]
            same_emotion = frame.get("emotion") == prev.get("emotion")

            if gap <= 3.0 and same_emotion:
                current_frames.append(frame)
            else:
                scenes.append(self._build_scene(current_frames))
                current_frames = [frame]

        # 最后一个场景
        scenes.append(self._build_scene(current_frames))
        return scenes

    @staticmethod
    def _build_scene(frames: list[dict]) -> dict:
        """从帧列表构建单个场景字典。"""
        scores = [f.get("highlight_score", 0) for f in frames]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        rep = frames[len(frames) // 2]  # 取中间帧作为代表
        return {
            "start": frames[0]["timestamp"],
            "end": frames[-1]["timestamp"],
            "scene_desc": rep.get("scene_desc", ""),
            "main_subject": rep.get("main_subject", ""),
            "action": rep.get("action", ""),
            "emotion": rep.get("emotion", "neutral"),
            "avg_highlight_score": round(avg_score, 2),
            "has_cut_point": any(f.get("cut_point", False) for f in frames),
            "frame_count": len(frames),
        }


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
            config = yaml.safe_load(f)

        from utils.llm_client import create_clients
        vl_client, _ = create_clients(config)

        try:
            stage1 = VideoUnderstanding(vl_client, config)
            scenes = await stage1.analyze_video(sys.argv[1])
            for i, scene in enumerate(scenes, 1):
                print(
                    f"场景 {i:02d} [{scene['start']:.1f}s - {scene['end']:.1f}s]  "
                    f"亮点分: {scene['avg_highlight_score']}  "
                    f"{scene['scene_desc'][:50]}"
                )
        finally:
            await vl_client.close()

    asyncio.run(_test())
