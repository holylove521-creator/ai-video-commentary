# -*- coding: utf-8 -*-
"""
ASR 语音识别阶段（faster-whisper 封装）。

本模块作为兼容入口保留；推荐直接使用
``pipeline.dialogue_stage.DialogueStage``，后者支持外挂 SRT 优先 + ASR 回退。
"""
from typing import Any, Dict, List

from pipeline.dialogue_stage import (
    extract_audio_track,
    transcribe_with_whisper,
)


class ASRStage:
    """faster-whisper 转写器（薄封装，兼容旧 API）。"""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """转写音频，返回 ``[{start,end,text}, ...]``。"""
        lines = transcribe_with_whisper(
            audio_path,
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return [{"start": d.start, "end": d.end, "text": d.text} for d in lines]


__all__ = ["ASRStage", "extract_audio_track", "transcribe_with_whisper"]
