"""
Stage 1: 视频理解（电影解说版）

新版流水线（专为长片电影解说设计）：

::

    [A] FrameExtractor.extract_thumbnails_nvdec  → thumbs (640x360, 2fps)
    [B] SceneDetector.detect_shots               → shots (~800-1500 个)
    [C] _aggregate_shots_to_scenes               → scenes (~150-250 个)
        + 把 DialogueLine 按时间轴 attach 到对应 scene
    [D] VL-7B 粗筛：每 scene 1 张代表帧（720p）+ 对白注入
        → {plot_role, importance, action, subjects, ...}
    [E] VL-32B 精分 top scenes（importance 加权 + 含对白优先）
        → {visual_desc, character_actions, mood, narrative_purpose}
    [F] 输出：list[Scene]（含 visual_desc/dialogue/importance）

向 Stage 2 暴露的两条数据通道：
- ``analyze_video(video_path, dialogue)`` 返回 list[Scene]（推荐）
- 同时把 Scene 折叠成兼容旧接口的 list[EventBlock]（``to_event_blocks``）
  以便旧代码继续工作。

关键改进:
1. 解码走 NVDEC，4K 不再卡死。
2. **每次 VL 调用 prompt 中注入该 scene 的对白文本**，VL 输出的 visual_desc
   能"对上戏"——这是连贯解说的基础。
3. shot→scene 聚合避免 VL 重复处理同一场景的多个 shot。
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm

from utils.frame_extractor import FrameExtractor
from utils.llm_client import LlamaCppClient
from utils.scene_detector import SceneDetector
from pipeline.schema import (
    DialogueLine,
    EventBlock,
    Scene,
    Shot,
)


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

_PROMPT_FAST_SYS = (
    "你是电影分析助手。基于一张代表帧画面与该场景的台词，输出紧凑 JSON。"
    "不要任何解释，只返回 JSON。"
)

_PROMPT_FAST_USER = """\
该场景对白（可能为空）：
\"\"\"{dialogue}\"\"\"

请返回 JSON：
{{
  "plot_role": "setup|conflict|twist|climax|resolution|filler",
  "subjects": "画面主体（如：男主/女主/反派/群众/物体）",
  "action": "正在发生什么（一句话）",
  "has_face": true|false,
  "importance": <0-10 整数，10=对剧情极关键>
}}
"""

_PROMPT_DEEP_SYS = (
    "你是资深电影解说编剧。结合画面与台词，写出**用于解说脚本撰写**的精炼场景描述。"
    "不要剧情评价，只描述客观信息。仅返回 JSON。"
)

_PROMPT_DEEP_USER = """\
该场景对白（可能为空）：
\"\"\"{dialogue}\"\"\"

请返回 JSON：
{{
  "visual_desc": "画面描述（人物外形/场景/镜头氛围，2-3句）",
  "character_actions": "人物在做什么（含台词意图，1-2句）",
  "mood": "情绪基调（如：紧张/温情/悬疑/搞笑/悲伤）",
  "narrative_purpose": "本场景在剧情中的作用（铺垫/冲突/反转/高潮/收束）"
}}
"""


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class VideoUnderstanding:
    """两阶段视频理解（NVDEC + Shot 聚合 + 对白注入）。"""

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

        self._fps_sample: float = float(video_cfg.get("fps_sample", 1.0))
        self._thumb_fps: float = float(video_cfg.get("thumb_fps", 2.0))
        self._frame_quality: int = int(video_cfg.get("frame_quality", 85))
        self._fast_resize: tuple[int, int] = tuple(video_cfg.get("fast_resize", [1280, 720]))
        self._deep_resize: tuple[int, int] = tuple(video_cfg.get("deep_resize", [1920, 1080]))
        self._max_concurrent: int = int(video_cfg.get(
            "max_concurrent_frames",
            video_cfg.get("max_concurrent", 8),
        ))
        self._max_concurrent_deep: int = int(video_cfg.get("max_concurrent_deep", 2))

        self._scene_detection_enabled: bool = bool(scene_cfg.get("enabled", True))
        self._scene_threshold: float = float(scene_cfg.get("threshold", 27.0))
        self._min_scene_len: int = int(scene_cfg.get("min_scene_len", 12))
        self._min_scene_gap: float = float(scene_cfg.get("min_scene_gap", 0.8))
        self._top_score_threshold: int = int(scene_cfg.get("top_score_threshold", 6))

        # Scene 聚合参数
        self._scene_merge_max_gap: float = float(scene_cfg.get("scene_merge_max_gap", 3.0))
        self._scene_target_min_dur: float = float(scene_cfg.get("scene_target_min_dur", 6.0))
        self._scene_target_max_dur: float = float(scene_cfg.get("scene_target_max_dur", 90.0))

        temp_dir = paths_cfg.get("temp_dir", "/tmp/ai_video_tmp")
        self._temp_dir = Path(temp_dir)
        self._temp_frames_dir = str(self._temp_dir / "frames")
        self._scene_repr_dir = self._temp_dir / "scene_repr"
        self._scene_repr_dir.mkdir(parents=True, exist_ok=True)

        self._extractor = FrameExtractor(temp_dir=self._temp_frames_dir)
        self._detector = SceneDetector(
            threshold=self._scene_threshold,
            min_scene_len=self._min_scene_len,
            min_scene_gap=self._min_scene_gap,
        )

    # ------------------------------------------------------------------
    # 公开主入口
    # ------------------------------------------------------------------

    async def analyze_video(
        self,
        video_path: str,
        fps_sample: Optional[float] = None,
        dialogue: Optional[list[DialogueLine]] = None,
    ) -> list[EventBlock]:
        """端到端分析视频，返回 EventBlock 列表（兼容旧 Stage 2 调用）。

        Args:
            video_path: 输入视频路径。
            fps_sample: 仅 legacy 模式使用；新模式忽略。
            dialogue:   可选对白列表（来自 ``DialogueStage``）；为 None 时
                        VL prompt 不注入台词。

        Returns:
            EventBlock 列表（按时间排序）。原始 Scene 列表通过
            ``self.last_scenes`` 暴露给上层。
        """
        scenes = await self.analyze_to_scenes(video_path, dialogue=dialogue)
        self.last_scenes: list[Scene] = scenes
        return self.scenes_to_event_blocks(scenes)

    async def analyze_to_scenes(
        self,
        video_path: str,
        dialogue: Optional[list[DialogueLine]] = None,
    ) -> list[Scene]:
        """新版主流程：返回完整的 Scene 列表。"""
        if not self._scene_detection_enabled:
            logger.warning(
                "[Stage1] scene_detection.enabled=false，已弃用；强制走新流水线"
            )

        t0 = time.time()

        # ── A. NVDEC 抽缩略图 ──
        logger.info("[Stage1] A. NVDEC 抽缩略图…")
        ta = time.time()
        thumbs = self._extractor.extract_thumbnails_nvdec(
            video_path, fps=self._thumb_fps, size=(640, 360),
        )
        logger.info(f"[Stage1] A 完成: {len(thumbs)} 张  ({time.time()-ta:.1f}s)")

        # ── B. Shot 切分 ──
        tb = time.time()
        shots = self._detector.detect_shots(thumbs, thumbs_fps=self._thumb_fps)
        if not shots:
            logger.warning("[Stage1] 未检测到 shot，返回空场景列表")
            return []
        logger.info(f"[Stage1] B 完成: {len(shots)} shot  ({time.time()-tb:.1f}s)")

        # ── C. Shot → Scene 聚合 + 对白对齐 ──
        tc = time.time()
        scenes = self._aggregate_shots_to_scenes(shots, dialogue or [])
        # 为每个 scene 选代表帧（取 shot 的 repr_frame_path 中位的）
        self._assign_scene_repr_frames(scenes, shots)
        logger.info(
            f"[Stage1] C 完成: {len(scenes)} scene  ({time.time()-tc:.1f}s)"
        )

        # ── D. VL-7B 粗筛 ──
        td = time.time()
        logger.info(
            f"[Stage1] D. VL-7B 粗筛 {len(scenes)} scene "
            f"(并发={self._max_concurrent})…"
        )
        await self._batch_vl(
            scenes, client=self._fast_client,
            sys_prompt=_PROMPT_FAST_SYS, user_template=_PROMPT_FAST_USER,
            max_tokens=128, semaphore=self._max_concurrent,
            resize=self._fast_resize, fields_apply=self._apply_fast_result,
            desc="粗筛",
        )
        logger.info(f"[Stage1] D 完成  ({time.time()-td:.1f}s)")

        # ── E. VL-32B 精分高重要性 + 含对白 scene ──
        te = time.time()
        top = self._select_top_scenes(scenes)
        logger.info(
            f"[Stage1] E. VL-32B 精分 {len(top)}/{len(scenes)} scene "
            f"(并发={self._max_concurrent_deep})…"
        )
        if top:
            await self._batch_vl(
                top, client=self._client,
                sys_prompt=_PROMPT_DEEP_SYS, user_template=_PROMPT_DEEP_USER,
                max_tokens=320, semaphore=self._max_concurrent_deep,
                resize=self._deep_resize, fields_apply=self._apply_deep_result,
                desc="精分",
            )
        logger.info(f"[Stage1] E 完成  ({time.time()-te:.1f}s)")

        total = time.time() - t0
        logger.success(
            f"[Stage1] 全部完成  scene={len(scenes)} 总耗时={total/60:.1f}min"
        )
        return scenes

    # ------------------------------------------------------------------
    # Shot → Scene 聚合
    # ------------------------------------------------------------------

    def _aggregate_shots_to_scenes(
        self,
        shots: list[Shot],
        dialogue: list[DialogueLine],
    ) -> list[Scene]:
        """把短 shot 合并为叙事 scene，附加对白。

        合并规则：
        - 相邻 shot 间隔 ≤ ``scene_merge_max_gap`` 且当前 scene 时长
          < ``scene_target_max_dur`` → 合并
        - 当前 scene 时长 < ``scene_target_min_dur`` 强制合并下一个
        """
        scenes: list[Scene] = []
        if not shots:
            return scenes

        cur_start = shots[0].start
        cur_end = shots[0].end
        cur_shot_ids = [shots[0].shot_id]
        cur_repr = shots[0].repr_frame_path

        def flush():
            nonlocal cur_start, cur_end, cur_shot_ids, cur_repr
            scenes.append(Scene(
                scene_id=len(scenes),
                start=round(cur_start, 3),
                end=round(cur_end, 3),
                shot_ids=cur_shot_ids[:],
                repr_frame_path=cur_repr,
            ))

        for sh in shots[1:]:
            gap = sh.start - cur_end
            cur_dur = cur_end - cur_start
            if (
                (gap <= self._scene_merge_max_gap and cur_dur < self._scene_target_max_dur)
                or cur_dur < self._scene_target_min_dur
            ):
                cur_end = sh.end
                cur_shot_ids.append(sh.shot_id)
            else:
                flush()
                cur_start, cur_end = sh.start, sh.end
                cur_shot_ids = [sh.shot_id]
                cur_repr = sh.repr_frame_path
        flush()

        # 把对白按时间 attach 到 scene
        scenes_sorted = scenes  # 已按 shot 顺序生成
        di = 0
        for sc in scenes_sorted:
            while di < len(dialogue) and dialogue[di].end <= sc.start:
                di += 1
            j = di
            while j < len(dialogue) and dialogue[j].start < sc.end:
                # 完全在 scene 内或与 scene 重叠
                sc.dialogue.append(dialogue[j])
                j += 1

        logger.info(
            f"[Stage1] Shot→Scene: shots={len(shots)} → scenes={len(scenes)}; "
            f"对白行总数={len(dialogue)}"
        )
        return scenes

    def _assign_scene_repr_frames(
        self,
        scenes: list[Scene],
        shots: list[Shot],
    ) -> None:
        """选每个 scene 中位 shot 的代表帧。"""
        shot_by_id = {s.shot_id: s for s in shots}
        for sc in scenes:
            if not sc.shot_ids:
                continue
            mid_id = sc.shot_ids[len(sc.shot_ids) // 2]
            mid_shot = shot_by_id.get(mid_id)
            if mid_shot and mid_shot.repr_frame_path:
                sc.repr_frame_path = mid_shot.repr_frame_path

    # ------------------------------------------------------------------
    # Top scene 选择（喂 VL-32B）
    # ------------------------------------------------------------------

    def _select_top_scenes(self, scenes: list[Scene]) -> list[Scene]:
        if not scenes:
            return []
        thr = self._top_score_threshold

        def keep(s: Scene) -> bool:
            score = s.importance
            # 含对白的略加权（对白通常承载剧情）
            if s.dialogue:
                score += 1.5
            # plot_role 是关键节点也保留
            if s.plot_role in {"conflict", "twist", "climax", "resolution"}:
                score += 1.0
            return score >= thr

        # 至少保留 30% 防止漏掉关键
        chosen = [s for s in scenes if keep(s)]
        min_keep = max(1, int(len(scenes) * 0.3))
        if len(chosen) < min_keep:
            chosen = sorted(
                scenes,
                key=lambda s: (s.importance + (1.5 if s.dialogue else 0.0)),
                reverse=True,
            )[:min_keep]
        return sorted(chosen, key=lambda s: s.start)

    # ------------------------------------------------------------------
    # VL 调用
    # ------------------------------------------------------------------

    async def _batch_vl(
        self,
        scenes: list[Scene],
        client: LlamaCppClient,
        sys_prompt: str,
        user_template: str,
        max_tokens: int,
        semaphore: int,
        resize: tuple[int, int],
        fields_apply,
        desc: str,
    ) -> None:
        sem = asyncio.Semaphore(semaphore)

        async def _one(sc: Scene):
            async with sem:
                try:
                    parsed = await self._call_vl_for_scene(
                        sc, client, sys_prompt, user_template,
                        max_tokens=max_tokens, resize=resize,
                    )
                    fields_apply(sc, parsed)
                except Exception as exc:
                    logger.warning(
                        f"[Stage1] scene {sc.scene_id} VL 调用失败: {exc}"
                    )

        tasks = [_one(s) for s in scenes if s.repr_frame_path]
        for coro in async_tqdm.as_completed(tasks, desc=desc, total=len(tasks)):
            await coro

    async def _call_vl_for_scene(
        self,
        scene: Scene,
        client: LlamaCppClient,
        sys_prompt: str,
        user_template: str,
        max_tokens: int,
        resize: tuple[int, int],
    ) -> dict:
        """对单个 scene 调用 VL：resize 代表帧 → 注入对白 → 调用 → 解析 JSON。"""
        if not scene.repr_frame_path or not Path(scene.repr_frame_path).exists():
            raise FileNotFoundError(scene.repr_frame_path or "<none>")

        img = cv2.imread(scene.repr_frame_path)
        if img is None:
            raise ValueError(f"无法读取代表帧: {scene.repr_frame_path}")
        if img.shape[:2] != (resize[1], resize[0]):
            img = cv2.resize(img, resize)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        dialogue_text = scene.dialogue_text(max_chars=200) or "（无台词）"
        user_msg = user_template.format(dialogue=dialogue_text)

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {"type": "text", "text": user_msg},
                ],
            },
        ]
        raw = await client.chat(messages, temperature=0.1, max_tokens=max_tokens)
        return self._safe_json(raw)

    # ------------------------------------------------------------------
    # JSON 容错
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # 把 VL 结果写回 Scene
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_fast_result(scene: Scene, parsed: dict) -> None:
        if not parsed:
            return
        try:
            scene.plot_role = str(parsed.get("plot_role") or scene.plot_role or "filler")
            scene.importance = float(parsed.get("importance", 0))
            scene.extra.update({
                "subjects": parsed.get("subjects", ""),
                "action": parsed.get("action", ""),
                "has_face": bool(parsed.get("has_face", False)),
            })
        except Exception:
            scene.extra["fast_parse_error"] = True

    @staticmethod
    def _apply_deep_result(scene: Scene, parsed: dict) -> None:
        if not parsed:
            return
        if "visual_desc" in parsed:
            scene.visual_desc = str(parsed["visual_desc"])
        scene.extra.update({
            "character_actions": parsed.get("character_actions", ""),
            "mood": parsed.get("mood", ""),
            "narrative_purpose": parsed.get("narrative_purpose", ""),
        })

    # ------------------------------------------------------------------
    # Scene → EventBlock（兼容旧 Stage 2）
    # ------------------------------------------------------------------

    @staticmethod
    def scenes_to_event_blocks(scenes: list[Scene]) -> list[EventBlock]:
        out: list[EventBlock] = []
        for sc in scenes:
            summary = sc.visual_desc or sc.extra.get("action", "") or "（无描述）"
            tags: list[str] = []
            mood = sc.extra.get("mood")
            if mood:
                tags.append(str(mood))
            if sc.plot_role:
                tags.append(sc.plot_role)
            out.append(EventBlock(
                start_time=sc.start,
                end_time=sc.end,
                type=sc.plot_role or "scene",
                summary=summary,
                characters=[],
                asr_transcript=sc.dialogue_text(max_chars=10_000) or None,
                visual_tags=tags,
                extra={
                    "scene_id": sc.scene_id,
                    "importance": sc.importance,
                    "narrative_purpose": sc.extra.get("narrative_purpose", ""),
                    "character_actions": sc.extra.get("character_actions", ""),
                    "subjects": sc.extra.get("subjects", ""),
                    "has_dialogue": bool(sc.dialogue),
                    "avg_highlight": sc.importance,
                },
            ))
        return out
