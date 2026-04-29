"""
Stage 2: 解说脚本生成

根据 Stage 1 输出的场景分析结果，调用 llama.cpp 文本大模型生成
带时间戳的结构化解说脚本，支持多种解说风格模板。

流程:
    1. 读取对应风格的 style_templates/*.yaml
    2. 精简场景数据（去掉冗余字段），构建 Prompt
    3. 调用 script_client 生成脚本 JSON
    4. 正则容错解析，校验时长合理性
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from utils.llm_client import LlamaCppClient


# ------------------------------------------------------------------
# 脚本生成 Prompt 模板
# ------------------------------------------------------------------

SCRIPT_PROMPT_TEMPLATE = """\
你是{persona}，风格：{tone}。

以下是视频的场景分析数据（JSON 列表）：
{scenes_json}

请根据上述场景为视频生成解说脚本，要求：
1. 每个场景对应 1-3 句解说词
2. 解说词要符合"{persona}"的风格特点
3. 语言生动，有感染力
4. 时间戳要与场景时间对应

请以 JSON 数组格式返回，每项包含：
- "start": 开始时间（秒，浮点数）
- "end": 结束时间（秒，浮点数）
- "text": 解说文本
- "emotion": 情感标签（excited/calm/serious/funny/tense/neutral 之一）

仅返回 JSON 数组，不要其他说明文字。
"""


class ScriptGenerator:
    """基于 LLM 的解说脚本生成器。

    Args:
        script_client: 连接脚本生成 llama-server 的异步客户端。
        style:         解说风格名称（game/sports/vlog/doc/comedy）。
        config:        全局配置字典。
    """

    def __init__(
        self,
        script_client: LlamaCppClient,
        style: str = "game",
        config: Optional[dict] = None,
    ) -> None:
        self._client = script_client
        self._style = style
        self._config = config or {}
        self._style_cfg = self._load_style(style)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    async def generate(self, scenes: list[dict]) -> list[dict]:
        """根据场景列表生成解说脚本。

        Args:
            scenes: Stage 1 输出的场景列表。

        Returns:
            脚本列表，每项格式::

                {"start": float, "end": float, "text": str, "emotion": str}

        Raises:
            RuntimeError: LLM 调用失败或返回无效 JSON。
        """
        if not scenes:
            logger.warning("[Stage2] 场景列表为空，返回空脚本")
            return []

        slim_scenes = self._slim_scenes(scenes)
        prompt = SCRIPT_PROMPT_TEMPLATE.format(
            persona=self._style_cfg.get("persona", "解说员"),
            tone=self._style_cfg.get("tone", "自然"),
            scenes_json=json.dumps(slim_scenes, ensure_ascii=False, indent=2),
        )

        system_prompt = self._style_cfg.get("system_prompt", "你是一名专业解说员。")
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"[Stage2] 开始生成脚本（风格: {self._style}，共 {len(scenes)} 个场景）")
        raw = await self._client.chat(messages, temperature=0.7, max_tokens=4096)
        script = self._parse_script(raw)
        script = self._validate_script(script, scenes)
        logger.success(f"[Stage2] 脚本生成完成，共 {len(script)} 段")
        return script

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _load_style(self, style: str) -> dict:
        """读取风格模板 YAML 文件。"""
        template_dir = Path("config/style_templates")
        style_file = template_dir / f"{style}.yaml"
        if not style_file.exists():
            logger.warning(
                f"[Stage2] 风格文件不存在: {style_file}，使用默认配置"
            )
            return {"persona": "解说员", "tone": "自然", "system_prompt": "你是一名专业解说员。"}
        with open(style_file, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _slim_scenes(scenes: list[dict]) -> list[dict]:
        """精简场景数据，仅保留 LLM 需要的字段以减少 token 消耗。"""
        keep = {"start", "end", "scene_desc", "action", "emotion", "avg_highlight_score"}
        return [
            {k: v for k, v in scene.items() if k in keep}
            for scene in scenes
        ]

    @staticmethod
    def _parse_script(raw: str) -> list[dict]:
        """解析 LLM 返回的脚本 JSON，含容错处理。

        尝试顺序：
        1. 直接 ``json.loads``
        2. 正则提取第一个 ``[...]`` 块后 ``json.loads``
        3. 返回空列表并记录警告

        Args:
            raw: LLM 返回的原始字符串。

        Returns:
            脚本列表，或空列表（解析失败时）。
        """
        # 方案 1：直接解析
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 方案 2：正则提取 JSON 数组
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning(f"[Stage2] 脚本 JSON 解析失败，原始内容前 300 字: {raw[:300]}")
        return []

    @staticmethod
    def _validate_script(
        script: list[dict], scenes: list[dict]
    ) -> list[dict]:
        """校验并修正脚本时间戳合理性。

        - 确保每段时长 ≥ 1 秒
        - 确保 start < end
        - 若脚本为空，按场景生成占位脚本

        Args:
            script: 原始脚本列表。
            scenes: 场景列表（用于生成占位脚本）。

        Returns:
            修正后的脚本列表。
        """
        if not script and scenes:
            logger.warning("[Stage2] 脚本为空，生成占位脚本")
            return [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s.get("scene_desc", ""),
                    "emotion": s.get("emotion", "neutral"),
                }
                for s in scenes
            ]

        validated: list[dict] = []
        for seg in script:
            try:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", start + 3))
                if end <= start:
                    end = start + 3.0
                if end - start < 1.0:
                    end = start + 1.0
                validated.append(
                    {
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": str(seg.get("text", "")),
                        "emotion": str(seg.get("emotion", "neutral")),
                    }
                )
            except (TypeError, ValueError) as exc:
                logger.warning(f"[Stage2] 跳过无效脚本段: {seg}  原因: {exc}")

        return validated


# ------------------------------------------------------------------
# 独立测试入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    async def _test():
        # 加载示例场景
        sample_scenes = [
            {
                "start": 0.0, "end": 5.0,
                "scene_desc": "玩家进入地图，四处张望",
                "action": "移动", "emotion": "calm",
                "avg_highlight_score": 3.0,
            },
            {
                "start": 5.0, "end": 12.0,
                "scene_desc": "突然遭遇敌方，激烈交火",
                "action": "战斗", "emotion": "excited",
                "avg_highlight_score": 8.5,
            },
        ]
        style = sys.argv[1] if len(sys.argv) > 1 else "game"

        import yaml
        with open("config/model_config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        from utils.llm_client import create_clients
        _, script_client = create_clients(config)

        try:
            gen = ScriptGenerator(script_client, style=style, config=config)
            script = await gen.generate(sample_scenes)
            for seg in script:
                print(
                    f"[{seg['start']:.1f}s - {seg['end']:.1f}s] "
                    f"({seg['emotion']}) {seg['text']}"
                )
        finally:
            await script_client.close()

    asyncio.run(_test())
