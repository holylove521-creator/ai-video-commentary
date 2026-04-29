# -*- coding: utf-8 -*-
"""
媒体探测工具：获取视频/音频基础信息（时长、分辨率、帧率等）
"""
import subprocess
from typing import Dict, Any


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    使用 ffprobe 获取视频基础信息。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return {
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(stream["duration"]),
        "fps": eval(stream["avg_frame_rate"]),
    }
