# -*- coding: utf-8 -*-
"""
ASR 语音识别阶段（faster-whisper 封装）
"""
from typing import List, Dict, Any

# 可选：import faster_whisper

class ASRStage:
    def __init__(self, model_path: str = "models/faster-whisper-large-v2", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        # self.model = faster_whisper.WhisperModel(model_path, device=device)

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        对音频文件进行转写，返回分段结果。
        """
        # segments, info = self.model.transcribe(audio_path)
        # return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        # 占位实现
        return [{"start": 0.0, "end": 1.0, "text": "（占位 ASR 结果）"}]
