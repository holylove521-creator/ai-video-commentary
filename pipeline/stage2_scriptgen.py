"""
Stage 2: 电影解说脚本生成（三段式）

设计目标：从一部完整电影的 ``Scene`` 列表 + 对白，生成一段 5-10 min、剧情连贯、
谷阿莫风格的中文解说脚本。

为什么三段式？
==============

旧实现把所有 scene 一次性塞给 LLM 一发出，2h 电影几百个 scene 会超 context、
没法做角色追踪、章节衔接稀烂。新流程拆为：

D1. **角色与剧情线抽取（Extract）**
    分块（每块 ~30 个 scene + 对应对白）滚动调用 LLM，维护
    ``{characters[], setting, plot_timeline[]}`` 状态，最后合并为整片剧情结构。

D2. **章节大纲生成（Outline）**
    输入完整剧情结构 + 全部 scene 摘要，输出 5-8 个章节，每章节包含
    ``{ch_id, narrative_role, target_seconds, source_scene_ids[], beat_summary}``，
    总时长强制对齐到目标（420 s ± 60 s）。

D3. **逐章节撰稿（Write）**
    每章节单独 LLM 调用（可并发），输入 = 该章节涉及 scene 的视觉+对白 +
    章节 beat + 上一章末两句解说锚 + 字数硬约束（``target_seconds × 4``）。
    输出 ``[{start, end, text, emotion, source_scene_id}]``。

最终输出 ``list[NarrationSegment]``。``start/end`` 仅作占位（粗估），真实剪辑
时长以 Stage 3 TTS 输出的实际音频时长为准。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from utils.llm_client import LlamaCppClient
from pipeline.schema import EventBlock, NarrationSegment


# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

_PROMPT_D1 = """\
你是电影编剧助手。下面是一段电影的场景列表（含画面描述与台词），按时间顺序排列。
请基于已有的"全片状态"，更新角色信息、世界观和剧情时间线。仅返回 JSON。

【已有全片状态（可能为空）】
{prev_state}

【本块场景（JSON 列表）】
{scenes_json}

请输出更新后的状态 JSON：
{{
  "characters": [
    {{"alias": "代称（男主/小哥/反派...）", "desc": "外形/职业/关键特征", "arc": "本块发生的变化"}}
  ],
  "setting": "世界观/背景（一段话）",
  "plot_timeline": [
    "时间点 1：发生了什么",
    "时间点 2：发生了什么"
  ]
}}

要求：
- characters 用代称，不引入电影里复杂的角色名。
- plot_timeline 每条 ≤ 40 字，按时间顺序。
- 如果本块没有新增信息，对应字段保留旧值。
"""

_PROMPT_D2 = """\
你是"谷阿莫式"电影解说编剧。

【全片剧情结构】
{plot_struct}

【全部场景摘要（JSON 列表，含 scene_id / start / end / desc / dialogue_brief）】
{scene_brief_json}

请规划解说视频的章节大纲。要求：
- 章节数 5-8 个。
- 总解说时长目标：{target_seconds} 秒（约 {target_chars} 字）。
- 每个章节都要绑定若干 source scene_id（可重复利用），保证整段剧情讲清楚。
- narrative_role 从 {chapter_roles_hint} 中选取或自定义类似项。
- 仅返回 JSON。

输出格式：
{{
  "chapters": [
    {{
      "ch_id": 1,
      "narrative_role": "开场钩子",
      "target_seconds": 30,
      "source_scene_ids": [0, 5, 7],
      "beat_summary": "本章节要讲的核心信息（一句话）"
    }}
  ]
}}

注意：所有 target_seconds 之和必须落在 [{tmin}, {tmax}] 区间内。
"""

_PROMPT_D3 = """\
你正在为一部电影写**谷阿莫式**中文解说稿，**只写本章节**。

【全片剧情结构（参考用，不要重复全讲）】
{plot_struct}

【上一章末尾两句解说（用于衔接，没有则为空）】
{prev_anchor}

【本章节信息】
- 章节作用：{narrative_role}
- 章节核心：{beat_summary}
- 目标时长：{target_seconds} 秒（**严格控制在 {min_chars}-{max_chars} 中文字之间**）

【本章节涉及的场景（按时间顺序）】
{scene_block_json}

要求：
1. 用极短的句子串起来，节奏要快。
2. 角色统一用代称（男主/女主/小哥/姐姐/反派/大叔/老头），**不要写电影里的真实角色名**。
3. 该剧透就剧透，重要因果一定讲清楚；不要留悬念。
4. 吐槽点 1-2 处即可，不要堆砌。
5. **不要换行**，把整章节解说放到一个或多个 JSON 段里。
6. 字数严格遵守上限。
7. 仅返回 JSON 数组：
[
  {{
    "text": "解说文本（不含换行）",
    "emotion": "excited|calm|serious|funny|tense|neutral",
    "source_scene_id": 0
  }}
]
"""


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class ScriptGenerator:
    """三段式电影解说脚本生成器。"""

    def __init__(
        self,
        script_client: LlamaCppClient,
        style: str = "movie",
        config: Optional[dict] = None,
        target_seconds: Optional[float] = None,
        movie_name: Optional[str] = None,
    ) -> None:
        self._client = script_client
        self._style = style
        self._config = config or {}
        self._style_cfg = self._load_style(style)
        self._movie_name = movie_name or "（未命名电影）"

        # 时长目标 / 字数估算
        default_target = float(self._style_cfg.get("default_target_seconds", 420))
        self._target_seconds: float = float(target_seconds or default_target)
        self._cps: float = float(self._style_cfg.get("chinese_chars_per_second", 4.0))

        # 大纲允许的总时长偏差（±60 s）
        self._target_min = max(60.0, self._target_seconds - 60.0)
        self._target_max = self._target_seconds + 60.0

        # 单段单调用并发
        self._d3_concurrent: int = int(self._config.get("script", {}).get("d3_concurrent", 2))

    # ------------------------------------------------------------------
    # 公开主入口
    # ------------------------------------------------------------------

    async def generate(self, scenes: list[EventBlock]) -> list[NarrationSegment]:
        """三段式生成解说。"""
        if not scenes:
            logger.warning("[Stage2] 场景列表为空，返回空脚本")
            return []

        logger.info(
            f"[Stage2] 电影解说生成: 风格={self._style}  "
            f"目标时长={self._target_seconds:.0f}s  scene 数={len(scenes)}"
        )

        # ── D1. 角色 + 剧情线抽取 ──
        plot_struct = await self._d1_extract_plot(scenes)
        logger.info(
            f"[Stage2] D1 完成: characters={len(plot_struct.get('characters', []))} "
            f"timeline={len(plot_struct.get('plot_timeline', []))}"
        )

        # ── D2. 章节大纲 ──
        outline = await self._d2_outline(scenes, plot_struct)
        logger.info(f"[Stage2] D2 完成: 章节数={len(outline)}")
        if not outline:
            logger.warning("[Stage2] 章节大纲为空，回退占位脚本")
            return self._placeholder(scenes)

        # ── D3. 逐章节撰稿 ──
        narration = await self._d3_write_all_chapters(outline, scenes, plot_struct)
        logger.success(f"[Stage2] D3 完成: 解说段数={len(narration)}")
        return narration

    # ------------------------------------------------------------------
    # D1. Extract
    # ------------------------------------------------------------------

    async def _d1_extract_plot(self, scenes: list[EventBlock]) -> dict:
        chunk_size = 30
        state: dict = {"characters": [], "setting": "", "plot_timeline": []}
        sys_prompt = self._style_cfg.get("system_prompt", "你是一名电影解说编剧。")

        chunks = [scenes[i:i + chunk_size] for i in range(0, len(scenes), chunk_size)]
        for ci, chunk in enumerate(chunks):
            scenes_json = json.dumps(
                [self._eb_to_extract_dict(eb) for eb in chunk],
                ensure_ascii=False,
            )
            user = _PROMPT_D1.format(
                prev_state=json.dumps(state, ensure_ascii=False),
                scenes_json=scenes_json,
            )
            messages = [
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user},
            ]
            try:
                raw = await self._client.chat(
                    messages, temperature=0.3, max_tokens=1024,
                )
                parsed = self._safe_json_obj(raw)
                if parsed:
                    state = self._merge_state(state, parsed)
                logger.info(f"[Stage2/D1] chunk {ci+1}/{len(chunks)} ok")
            except Exception as exc:
                logger.warning(f"[Stage2/D1] chunk {ci} 失败: {exc}")

        return state

    @staticmethod
    def _eb_to_extract_dict(eb: EventBlock) -> dict:
        return {
            "start": round(eb.start_time, 1),
            "end": round(eb.end_time, 1),
            "desc": eb.summary[:200] if eb.summary else "",
            "dialogue": (eb.asr_transcript or "")[:240],
            "subjects": eb.extra.get("subjects", "") if eb.extra else "",
        }

    @staticmethod
    def _merge_state(old: dict, new: dict) -> dict:
        out = {
            "characters": new.get("characters") or old.get("characters", []),
            "setting": new.get("setting") or old.get("setting", ""),
            "plot_timeline": (
                old.get("plot_timeline", []) + (new.get("plot_timeline") or [])
            ),
        }
        # 去重（基于 alias）
        seen, dedup = set(), []
        for c in out["characters"]:
            alias = (c or {}).get("alias")
            if alias and alias not in seen:
                seen.add(alias)
                dedup.append(c)
        out["characters"] = dedup
        return out

    # ------------------------------------------------------------------
    # D2. Outline
    # ------------------------------------------------------------------

    async def _d2_outline(
        self,
        scenes: list[EventBlock],
        plot_struct: dict,
    ) -> list[dict]:
        sys_prompt = self._style_cfg.get("system_prompt", "你是一名电影解说编剧。")
        scene_brief = [
            {
                "scene_id": eb.extra.get("scene_id", i) if eb.extra else i,
                "start": round(eb.start_time, 1),
                "end": round(eb.end_time, 1),
                "desc": (eb.summary or "")[:120],
                "dialogue_brief": (eb.asr_transcript or "")[:80],
                "importance": (eb.extra or {}).get("importance", 0),
            }
            for i, eb in enumerate(scenes)
        ]
        chapter_roles_hint = (
            self._style_cfg.get("chapter_roles")
            or ["开场钩子", "起", "承", "转", "合", "尾"]
        )
        target_chars = int(self._target_seconds * self._cps)
        user = _PROMPT_D2.format(
            plot_struct=json.dumps(plot_struct, ensure_ascii=False),
            scene_brief_json=json.dumps(scene_brief, ensure_ascii=False),
            target_seconds=int(self._target_seconds),
            target_chars=target_chars,
            chapter_roles_hint=json.dumps(chapter_roles_hint, ensure_ascii=False),
            tmin=int(self._target_min),
            tmax=int(self._target_max),
        )
        messages = [
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user},
        ]
        raw = await self._client.chat(
            messages, temperature=0.5, max_tokens=2048,
        )
        parsed = self._safe_json_obj(raw)
        chapters = (parsed or {}).get("chapters") or []
        chapters = self._normalize_outline(chapters, scenes)
        return chapters

    def _normalize_outline(
        self,
        chapters: list[dict],
        scenes: list[EventBlock],
    ) -> list[dict]:
        """校验并修正章节大纲（强制总时长落在区间内）。"""
        if not chapters:
            return []
        # 把 source_scene_ids 中无效项过滤
        valid_ids = {
            (eb.extra or {}).get("scene_id", i)
            for i, eb in enumerate(scenes)
        }
        for ch in chapters:
            ids = ch.get("source_scene_ids") or []
            ch["source_scene_ids"] = [
                int(x) for x in ids if isinstance(x, (int, float)) and int(x) in valid_ids
            ]
            ch["target_seconds"] = max(8.0, float(ch.get("target_seconds", 30)))

        # 总时长归一
        total = sum(ch["target_seconds"] for ch in chapters)
        if total <= 0:
            return []
        if not (self._target_min <= total <= self._target_max):
            scale = self._target_seconds / total
            for ch in chapters:
                ch["target_seconds"] = round(ch["target_seconds"] * scale, 1)
            logger.info(
                f"[Stage2/D2] 总时长 {total:.0f}s → 归一到 {self._target_seconds:.0f}s"
            )
        # 给空 source_scene_ids 的章节补一个默认（按时间均分）
        for i, ch in enumerate(chapters):
            if not ch["source_scene_ids"] and scenes:
                idx = min(len(scenes) - 1, int(i / max(1, len(chapters) - 1) * (len(scenes) - 1)))
                eb = scenes[idx]
                ch["source_scene_ids"] = [(eb.extra or {}).get("scene_id", idx)]
        return chapters

    # ------------------------------------------------------------------
    # D3. Write each chapter
    # ------------------------------------------------------------------

    async def _d3_write_all_chapters(
        self,
        outline: list[dict],
        scenes: list[EventBlock],
        plot_struct: dict,
    ) -> list[NarrationSegment]:
        scene_by_id: dict[int, EventBlock] = {}
        for i, eb in enumerate(scenes):
            sid = (eb.extra or {}).get("scene_id", i)
            scene_by_id[int(sid)] = eb

        results: list[Optional[list[dict]]] = [None] * len(outline)
        sem = asyncio.Semaphore(self._d3_concurrent)

        async def _one(idx: int, ch: dict, prev_anchor: str):
            async with sem:
                items = await self._d3_write_chapter(
                    ch, scene_by_id, plot_struct, prev_anchor
                )
                results[idx] = items

        # 串行触发以便 prev_anchor 衔接（用上一章节最后两句作为锚）；
        # 但同一时刻最多 N 个 LLM 调用排队
        prev_anchor = ""
        # 第一遍只算 prev_anchor 占位（章节顺序撰稿）
        # 简化实现：顺序 await，避免 anchor 错乱（D3 速度本身不是瓶颈）
        for idx, ch in enumerate(outline):
            try:
                items = await self._d3_write_chapter(
                    ch, scene_by_id, plot_struct, prev_anchor
                )
            except Exception as exc:
                logger.warning(f"[Stage2/D3] 章节 {idx} 失败: {exc}")
                items = []
            results[idx] = items
            if items:
                # 用最后一条 text 末尾 30 字作锚
                last_text = items[-1].get("text", "")
                prev_anchor = last_text[-30:]

        # 拼合并赋时间戳（粗估，真实时长由 TTS 决定）
        narration: list[NarrationSegment] = []
        cursor = 0.0
        for ci, items in enumerate(results):
            ch = outline[ci]
            if not items:
                continue
            ch_seconds = float(ch.get("target_seconds", 30))
            ch_start = cursor
            # 按字数比例分配时间
            total_chars = max(1, sum(len(it.get("text", "")) for it in items))
            for it in items:
                text = (it.get("text") or "").strip()
                if not text:
                    continue
                dur = ch_seconds * len(text) / total_chars
                start = round(cursor, 2)
                end = round(cursor + dur, 2)
                narration.append(NarrationSegment(
                    text=text,
                    event_block_index=int(it.get("source_scene_id", -1)),
                    speaker="解说",
                    style=self._style,
                    start_time=start,
                    end_time=end,
                    extra={
                        "emotion": it.get("emotion", "neutral"),
                        "chapter_id": ch.get("ch_id", ci + 1),
                        "narrative_role": ch.get("narrative_role", ""),
                        "source_scene_id": it.get("source_scene_id"),
                    },
                ))
                cursor = end
            cursor = ch_start + ch_seconds  # 防止误差累积，强制对齐章节边界

        return narration

    async def _d3_write_chapter(
        self,
        ch: dict,
        scene_by_id: dict[int, EventBlock],
        plot_struct: dict,
        prev_anchor: str,
    ) -> list[dict]:
        sys_prompt = self._style_cfg.get("system_prompt", "你是一名电影解说员。")
        target_seconds = float(ch.get("target_seconds", 30))
        target_chars = int(target_seconds * self._cps)
        min_chars = max(8, int(target_chars * 0.7))
        max_chars = max(min_chars + 4, int(target_chars * 1.15))

        scene_block = []
        for sid in ch.get("source_scene_ids", []):
            eb = scene_by_id.get(int(sid))
            if not eb:
                continue
            scene_block.append({
                "scene_id": (eb.extra or {}).get("scene_id", sid),
                "start": round(eb.start_time, 1),
                "end": round(eb.end_time, 1),
                "visual": (eb.summary or "")[:220],
                "dialogue": (eb.asr_transcript or "")[:200],
                "purpose": (eb.extra or {}).get("narrative_purpose", ""),
            })
        if not scene_block:
            return []

        user = _PROMPT_D3.format(
            plot_struct=json.dumps(plot_struct, ensure_ascii=False)[:1200],
            prev_anchor=prev_anchor or "（无）",
            narrative_role=ch.get("narrative_role", ""),
            beat_summary=ch.get("beat_summary", ""),
            target_seconds=int(target_seconds),
            min_chars=min_chars,
            max_chars=max_chars,
            scene_block_json=json.dumps(scene_block, ensure_ascii=False),
        )
        messages = [
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user", "content": user},
        ]
        raw = await self._client.chat(
            messages, temperature=0.75,
            max_tokens=min(2048, max_chars * 3 + 256),
        )
        items = self._safe_json_arr(raw)
        # 合并 text 并按字数硬截断
        items = self._enforce_chars(items, max_chars)
        return items

    @staticmethod
    def _enforce_chars(items: list[dict], max_chars: int) -> list[dict]:
        out: list[dict] = []
        budget = max_chars
        for it in items:
            text = re.sub(r"\s+", "", str(it.get("text", "")))
            if not text:
                continue
            if len(text) > budget:
                text = text[:budget]
            it["text"] = text
            budget -= len(text)
            out.append(it)
            if budget <= 0:
                break
        return out

    # ------------------------------------------------------------------
    # 占位
    # ------------------------------------------------------------------

    def _placeholder(self, scenes: list[EventBlock]) -> list[NarrationSegment]:
        out: list[NarrationSegment] = []
        for i, eb in enumerate(scenes[:8]):
            out.append(NarrationSegment(
                text=(eb.summary or "")[:60] or "（占位解说）",
                event_block_index=i,
                speaker="解说",
                style=self._style,
                start_time=eb.start_time,
                end_time=eb.end_time,
                extra={"emotion": "neutral", "placeholder": True},
            ))
        return out

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _load_style(self, style: str) -> dict:
        template_dir = Path("config/style_templates")
        style_file = template_dir / f"{style}.yaml"
        if not style_file.exists():
            logger.warning(
                f"[Stage2] 风格文件不存在: {style_file}，使用默认配置"
            )
            return {"persona": "解说员", "tone": "自然", "system_prompt": "你是一名解说员。"}
        with open(style_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _safe_json_obj(raw: str) -> dict:
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                return d if isinstance(d, dict) else {}
            except Exception:
                pass
        return {}

    @staticmethod
    def _safe_json_arr(raw: str) -> list[dict]:
        try:
            d = json.loads(raw)
            if isinstance(d, list):
                return d
        except Exception:
            pass
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                if isinstance(d, list):
                    return d
            except Exception:
                pass
        return []
