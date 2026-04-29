# -*- coding: utf-8 -*-
"""
核心数据结构定义：EventBlock、ChapterPlan、NarrationSegment、MixSegment
所有流水线阶段统一的数据边界
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class DialogueLine:
    """单条对白时间轴（来自外挂 SRT 或 ASR）。"""
    start: float
    end: float
    text: str
    source: str = "srt"          # "srt" | "asr"
    speaker: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return DialogueLine(**d)


@dataclass
class Shot:
    """PySceneDetect 输出的 shot（单镜头）。"""
    shot_id: int
    start: float
    end: float
    repr_frame_path: Optional[str] = None  # 代表帧 JPEG 路径

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Shot(**d)


@dataclass
class Scene:
    """叙事 scene = 多个 shot 聚合 + 对应对白。"""
    scene_id: int
    start: float
    end: float
    shot_ids: List[int] = field(default_factory=list)
    repr_frame_path: Optional[str] = None
    dialogue: List[DialogueLine] = field(default_factory=list)
    # VL 输出（粗筛 / 精分逐步填充）
    plot_role: Optional[str] = None        # setup|conflict|twist|climax|resolution|filler
    importance: float = 0.0                # 0-10
    visual_desc: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def dialogue_text(self, max_chars: int = 200) -> str:
        """合并对白文本，超长截断（避免撑爆 VL prompt）。"""
        joined = " ".join(d.text.strip() for d in self.dialogue if d.text)
        if len(joined) > max_chars:
            joined = joined[:max_chars] + "…"
        return joined

    def to_dict(self):
        d = asdict(self)
        d["dialogue"] = [x.to_dict() for x in self.dialogue]
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["dialogue"] = [DialogueLine.from_dict(x) for x in d.get("dialogue", [])]
        return Scene(**d)


@dataclass
class EventBlock:
    """
    表示一个有语义的连续视频片段（事件/场景）
    """
    start_time: float  # 起始时间（秒）
    end_time: float    # 结束时间（秒）
    type: str          # 事件类型（如：动作、对话、转场等）
    summary: str       # 事件摘要/场景描述
    characters: List[str] = field(default_factory=list)  # 主要人物
    asr_transcript: Optional[str] = None                # ASR转写文本
    visual_tags: List[str] = field(default_factory=list) # 视觉标签（如：夜景、爆炸）
    extra: Dict[str, Any] = field(default_factory=dict)  # 其他扩展信息

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return EventBlock(**d)

@dataclass
class ChapterPlan:
    """
    一组EventBlock的高层叙事规划（如章节/故事段）
    """
    chapter_index: int
    event_blocks: List[EventBlock]
    intent: str                  # 叙事意图/主线
    style: str                   # 解说风格（如：幽默、科普）
    memory: Optional[str] = None # 上下文记忆/剧情承接
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['event_blocks'] = [eb.to_dict() for eb in self.event_blocks]
        return d

    @staticmethod
    def from_dict(d):
        d['event_blocks'] = [EventBlock.from_dict(eb) for eb in d['event_blocks']]
        return ChapterPlan(**d)

@dataclass
class NarrationSegment:
    """
    单个解说文本单元（用于TTS合成）
    """
    text: str
    event_block_index: int       # 关联的EventBlock索引
    speaker: str = "旁白"         # 说话人
    style: Optional[str] = None  # 风格
    start_time: Optional[float] = None # 推荐起始时间
    end_time: Optional[float] = None   # 推荐结束时间
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return NarrationSegment(**d)

@dataclass
class MixSegment:
    """
    最终混剪的媒体片段（音频、视频、字幕）
    """
    start_time: float
    end_time: float
    narration_audio: Optional[str] = None  # 解说音频文件路径
    subtitle_file: Optional[str] = None    # 字幕文件路径
    video_file: Optional[str] = None       # 视频片段路径
    instructions: Optional[str] = None     # 混剪指令/备注
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return MixSegment(**d)

# 序列化/反序列化工具

def schema_from_json(cls, s: str):
    return cls.from_dict(json.loads(s))

def schema_list_to_json(obj_list) -> str:
    return json.dumps([o.to_dict() for o in obj_list], ensure_ascii=False, indent=2)

def schema_list_from_json(cls, s: str):
    arr = json.loads(s)
    return [cls.from_dict(x) for x in arr]
