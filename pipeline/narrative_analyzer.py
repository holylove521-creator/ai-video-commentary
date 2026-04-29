"""
叙事结构分析模块
负责分析视频场景的因果关系、剧情主线和场景分类，
为连贯性优先剪辑提供数据支撑。
"""

import json
import re

from loguru import logger

from utils.llm_client import LlamaCppClient

# 叙事结构分析 Prompt
NARRATIVE_ANALYSIS_PROMPT = """
你是一名专业视频剪辑师。以下是视频的所有场景列表（JSON格式）：

{scenes_json}

请完成以下任务：
1. 识别视频的【核心叙事主线】（人物/事件/目标）
2. 将所有场景按索引归类为：
   - "必要": 剧情推进不可缺少（删除会让观众看不懂）
   - "增色": 可保留但不影响理解
   - "可删": 重复或无关内容
3. 识别【关键因果关系】（哪个场景是哪个场景的原因/结果）
4. 标记【转折点】（剧情发生重要变化的时刻）
5. 识别【四幕结构】（开端/发展/高潮/结局对应哪些场景索引）

严格输出以下 JSON 格式，不要有任何其他文字：
{
  "narrative_core": "核心叙事主线一句话描述",
  "plot_structure": {
    "开端": [场景索引列表],
    "发展": [场景索引列表],
    "高潮": [场景索引列表],
    "结局": [场景索引列表]
  },
  "causal_chains": [
    {"cause": 场景索引, "effect": 场景索引, "relation": "因果关系描述"}
  ],
  "scenes_classification": [
    {"index": 场景索引, "type": "必要|增色|可删", "reason": "分类原因"}
  ],
  "turning_points": [场景索引列表]
}
"""

# 过渡解说生成 Prompt
TRANSITION_PROMPT = """
视频剪辑时从场景A直接跳转到场景B，中间跳过了约 {gap:.1f} 秒的内容。

场景A结尾描述：{desc_a}
场景B开头描述：{desc_b}

请生成一句自然流畅的过渡解说词（10-20字），
帮助观众理解这个跳跃，不能让人感到突兀。
只返回解说词文本，不要标点符号以外的任何内容。
"""

# 连贯性评分 Prompt
COHERENCE_CHECK_PROMPT = """
以下是剪辑后视频的解说脚本（按时间顺序）：

{script_json}

请从以下4个维度评估剧情连贯性（每项0-10分）：
1. 时间线清晰度：观众能否理解事件发生顺序
2. 因果逻辑：前后场景是否有合理的因果关系
3. 人物/主体连续性：主角/主体是否前后一致
4. 信息完整性：是否有关键信息缺失导致理解困难

严格输出以下 JSON 格式：
{
  "total_score": 综合评分(0-10的浮点数),
  "dimension_scores": {
    "时间线清晰度": 分数,
    "因果逻辑": 分数,
    "人物连续性": 分数,
    "信息完整性": 分数
  },
  "issues": [
    {
      "position": "第N段->第M段",
      "problem": "具体问题描述",
      "severity": "high|medium|low"
    }
  ],
  "suggestion": "总体修复建议"
}
"""


class NarrativeAnalyzer:
    """叙事结构分析器，保障剪辑连贯性"""

    def __init__(self, script_client: LlamaCppClient, coherence_threshold: float = 7.0) -> None:
        self.client = script_client
        self.coherence_threshold = coherence_threshold

    async def analyze_narrative(self, scenes: list[dict]) -> dict:
        """
        分析场景列表的叙事结构
        返回包含因果链、场景分类、转折点的结构化数据
        """
        # slim 化场景数据
        scenes_slim = [
            {
                "index": i,
                "start": s["start"],
                "end": s["end"],
                "summary": s.get("summary", ""),
                "highlight_score": round(s.get("avg_highlight", 5), 1),
                "emotion": (s["frames"][0].get("emotion", "calm") if s.get("frames") else "calm"),
            }
            for i, s in enumerate(scenes)
        ]

        prompt = NARRATIVE_ANALYSIS_PROMPT.format(
            scenes_json=json.dumps(scenes_slim, ensure_ascii=False, indent=2)
        )

        result = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )

        try:
            narrative = json.loads(result)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    narrative = json.loads(match.group())
                except json.JSONDecodeError:
                    narrative = None
            else:
                narrative = None

        if narrative is None:
            logger.warning("叙事分析 JSON 解析失败，使用降级策略（全部场景标记为必要）")
            narrative = self._fallback_narrative(scenes)

        logger.info(f"叙事主线: {narrative.get('narrative_core', '未识别')}")
        logger.info(f"转折点场景: {narrative.get('turning_points', [])}")
        return narrative

    def select_scenes_by_narrative(
        self,
        scenes: list[dict],
        narrative: dict,
        target_duration: float,
    ) -> list[dict]:
        """
        连贯性优先的场景选择算法：
        1. 必要场景强制保留（剧情骨架）
        2. 转折点强制保留
        3. 因果链保护（有 effect 必须有 cause）
        4. 剩余时长按 highlight_score 填充"增色"场景
        5. 按原始时间顺序排列
        """
        classifications = {
            item["index"]: item["type"]
            for item in narrative.get("scenes_classification", [])
        }

        must_keep: set[int] = set()

        # Step 1: 必要场景
        for idx, cls in classifications.items():
            if cls == "必要":
                must_keep.add(idx)

        # Step 2: 转折点
        for idx in narrative.get("turning_points", []):
            must_keep.add(idx)

        # Step 3: 因果链保护
        for chain in narrative.get("causal_chains", []):
            cause_idx = chain.get("cause")
            effect_idx = chain.get("effect")
            if effect_idx in must_keep and cause_idx is not None:
                must_keep.add(cause_idx)
                logger.debug(f"因果链保护: 场景{cause_idx} → 场景{effect_idx}")

        # Step 4: 按剩余时长填充增色场景
        def scene_duration(idx: int) -> float:
            if idx < len(scenes):
                s = scenes[idx]
                return s["end"] - s["start"]
            return 0.0

        must_duration = sum(scene_duration(i) for i in must_keep)
        remaining = max(0.0, target_duration - must_duration)

        optional = [
            (i, scenes[i].get("avg_highlight", 0))
            for i, cls in classifications.items()
            if cls == "增色" and i not in must_keep and i < len(scenes)
        ]
        optional.sort(key=lambda x: x[1], reverse=True)

        selected = set(must_keep)
        for idx, _ in optional:
            dur = scene_duration(idx)
            if remaining >= dur:
                selected.add(idx)
                remaining -= dur

        result = sorted([scenes[i] for i in selected if i < len(scenes)], key=lambda s: s["start"])
        logger.info(f"连贯性剪辑: 从 {len(scenes)} 个场景中选出 {len(result)} 个（必要:{len(must_keep)}）")
        return result

    async def generate_transition(self, scene_a: dict, scene_b: dict) -> str:
        """
        在两个跳跃场景之间生成过渡解说词
        跳跃小于 3 秒则返回空字符串
        """
        gap = scene_b["start"] - scene_a["end"]
        if gap < 3.0:
            return ""

        prompt = TRANSITION_PROMPT.format(
            gap=gap,
            desc_a=scene_a.get("summary", "上一个场景"),
            desc_b=scene_b.get("summary", "下一个场景"),
        )

        transition = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=60,
        )
        transition = transition.strip()
        logger.debug(f"生成过渡解说（跳跃{gap:.1f}s）: {transition}")
        return transition

    async def inject_transitions(self, script: list[dict]) -> list[dict]:
        """
        遍历脚本，在跳跃处注入过渡解说段
        """
        if len(script) < 2:
            return script

        enriched: list[dict] = [script[0]]
        for i in range(1, len(script)):
            prev = script[i - 1]
            curr = script[i]
            gap = curr["start"] - prev["end"]

            if gap >= 3.0:
                transition_text = await self.generate_transition(prev, curr)
                if transition_text:
                    # Use half the gap as transition duration (max 3 s) to avoid
                    # consuming too much of the gap with voiceover bridging.
                    trans_duration = min(gap * 0.5, 3.0)
                    enriched.append({
                        "start": prev["end"],
                        "end": prev["end"] + trans_duration,
                        "text": transition_text,
                        "emotion": "calm",
                        "duration_target": trans_duration,
                        "is_transition": True,
                    })

            enriched.append(curr)

        logger.info(f"过渡解说注入完成，脚本段数: {len(script)} → {len(enriched)}")
        return enriched

    async def coherence_check(self, script: list[dict]) -> dict:
        """
        对最终脚本做连贯性评分
        total_score < coherence_threshold 时记录警告并返回 issues
        """
        script_slim = [
            {"index": i, "start": s["start"], "end": s["end"], "text": s.get("text", "")}
            for i, s in enumerate(script)
        ]

        prompt = COHERENCE_CHECK_PROMPT.format(
            script_json=json.dumps(script_slim, ensure_ascii=False, indent=2)
        )

        result = await self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )

        try:
            score_data = json.loads(result)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                try:
                    score_data = json.loads(match.group())
                except json.JSONDecodeError:
                    score_data = {"total_score": 5.0, "issues": []}
            else:
                score_data = {"total_score": 5.0, "issues": []}

        total = score_data.get("total_score", 5.0)
        if total < self.coherence_threshold:
            logger.warning(
                f"连贯性评分 {total:.1f} 低于阈值 {self.coherence_threshold}，"
                f"存在 {len(score_data.get('issues', []))} 个问题"
            )
            for issue in score_data.get("issues", []):
                logger.warning(f"  [{issue.get('severity','?')}] {issue.get('position')}: {issue.get('problem')}")
        else:
            logger.info(f"连贯性评分: {total:.1f} ✅")

        return score_data

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _fallback_narrative(self, scenes: list[dict]) -> dict:
        """JSON 解析失败时的降级策略：所有场景标记为必要"""
        return {
            "narrative_core": "未能解析叙事结构，保留全部场景",
            "plot_structure": {"开端": [], "发展": [], "高潮": [], "结局": []},
            "causal_chains": [],
            "scenes_classification": [
                {"index": i, "type": "必要", "reason": "降级策略"} for i in range(len(scenes))
            ],
            "turning_points": [],
        }


if __name__ == "__main__":
    import asyncio

    async def _test() -> None:
        client = LlamaCppClient("http://localhost:8002")
        ok = await client.health_check()
        if not ok:
            print("script_server 未启动，跳过测试")
            return

        analyzer = NarrativeAnalyzer(client)
        mock_scenes = [
            {"start": 0, "end": 5, "summary": "主角出场，拿起武器", "avg_highlight": 6},
            {"start": 5, "end": 12, "summary": "敌人突然袭击", "avg_highlight": 9},
            {"start": 12, "end": 18, "summary": "主角躲避攻击", "avg_highlight": 7},
            {"start": 18, "end": 25, "summary": "主角反击成功", "avg_highlight": 10},
            {"start": 25, "end": 30, "summary": "战斗结束，主角获胜", "avg_highlight": 8},
        ]
        narrative = await analyzer.analyze_narrative(mock_scenes)
        print(json.dumps(narrative, ensure_ascii=False, indent=2))

        selected = analyzer.select_scenes_by_narrative(mock_scenes, narrative, target_duration=20.0)
        print(f"选出场景数: {len(selected)}")

        await client.close()

    asyncio.run(_test())
