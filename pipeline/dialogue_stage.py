"""
对白获取阶段：外挂 SRT 优先，无则 fallback 到 faster-whisper ASR。

输出统一的 DialogueLine 列表，按时间排序。

设计原则
========

1. **SRT 是 ground-truth**：用户提供 ``--srt`` 时直接解析，跳过 ASR。
2. **ASR fallback**：使用 faster-whisper large-v3，int8_float16 在 4090
   级显卡上 2h 16kHz 音频约 3-5 min。
3. **抽音轨**：用 ffmpeg + NVDEC 解复用，仅做单声道 16kHz 重采样。

主入口
------

::

    dialogue = DialogueStage(config).extract(
        video_path="movie.mkv",
        srt_path="movie.zh.srt",   # 可选
    )
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from loguru import logger

from pipeline.schema import DialogueLine


# ---------------------------------------------------------------------------
# SRT 解析（最小实现，避免新增 pysrt 依赖；同时能处理 BOM / CRLF / 多种时间格式）
# ---------------------------------------------------------------------------

_SRT_TIME = re.compile(
    r"(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2})[,.](?P<ms>\d{1,3})"
)


def _srt_time_to_seconds(token: str) -> float:
    m = _SRT_TIME.search(token)
    if not m:
        raise ValueError(f"无法解析 SRT 时间戳: {token!r}")
    return (
        int(m["h"]) * 3600
        + int(m["m"]) * 60
        + int(m["s"])
        + int(m["ms"]) / 1000.0
    )


def parse_srt(srt_path: str) -> List[DialogueLine]:
    """解析 SRT 文件，返回 DialogueLine 列表。

    容错点:
    - UTF-8 / UTF-8-BOM / GBK 编码自动尝试
    - CRLF / LF 混排
    - 缺少序号或空行的不规范文件

    Args:
        srt_path: SRT 文件路径。

    Returns:
        按时间排序的对白行列表。
    """
    raw: Optional[str] = None
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            raw = Path(srt_path).read_text(encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    if raw is None:
        raise RuntimeError(f"无法解码 SRT 文件: {srt_path} ({last_err})")

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw.strip())
    lines: List[DialogueLine] = []
    for block in blocks:
        rows = [r for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        # 找到第一条含 --> 的行作为时间戳
        time_row_idx = None
        for i, row in enumerate(rows):
            if "-->" in row:
                time_row_idx = i
                break
        if time_row_idx is None:
            continue
        try:
            t_start, t_end = rows[time_row_idx].split("-->")
            start = _srt_time_to_seconds(t_start.strip())
            end = _srt_time_to_seconds(t_end.strip())
        except Exception as exc:
            logger.debug(f"[SRT] 跳过无法解析的块: {rows[time_row_idx]!r} ({exc})")
            continue
        text = " ".join(rows[time_row_idx + 1:]).strip()
        # 去除常见的样式标签 <i> </i> {\an8} 等
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\{[^}]+\}", "", text)
        if not text:
            continue
        lines.append(DialogueLine(start=start, end=end, text=text, source="srt"))

    lines.sort(key=lambda x: x.start)
    logger.info(f"[Dialogue] SRT 解析完成: {len(lines)} 行 ← {srt_path}")
    return lines


# ---------------------------------------------------------------------------
# 音轨抽取（ffmpeg + NVDEC，单声道 16kHz）
# ---------------------------------------------------------------------------

def extract_audio_track(
    video_path: str,
    out_wav: str,
    use_cuda: bool = True,
) -> str:
    """从视频抽出 16kHz 单声道 wav，供 ASR 使用。

    Args:
        video_path: 输入视频路径。
        out_wav:    输出 wav 路径。
        use_cuda:   是否启用 NVDEC 解码加速；解码音频时影响有限，
                    仅在视频是 4K HEVC 时偶有收益。

    Returns:
        输出 wav 路径。
    """
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if use_cuda:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav),
    ]
    logger.info(f"[Dialogue] 抽取音轨 → {out_wav}")
    subprocess.run(cmd, check=True)
    return out_wav


# ---------------------------------------------------------------------------
# ASR (faster-whisper)
# ---------------------------------------------------------------------------

def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "int8_float16",
    language: Optional[str] = None,
) -> List[DialogueLine]:
    """用 faster-whisper 转写音频。

    ``compute_type`` 选 ``int8_float16`` 在 48GB 卡上速度/显存平衡最好。

    Args:
        audio_path:   wav 路径。
        model_size:   模型大小（large-v3 推荐）。
        device:       cuda | cpu | auto。
        compute_type: int8_float16 | float16 | int8。
        language:     强制语言（None 则自动检测）。

    Returns:
        DialogueLine 列表（source="asr"）。
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper 未安装，请: pip install faster-whisper"
        ) from exc

    logger.info(
        f"[Dialogue] 加载 faster-whisper {model_size} "
        f"(device={device}, compute={compute_type})"
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=True,
    )
    out: List[DialogueLine] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        out.append(DialogueLine(
            start=float(s.start),
            end=float(s.end),
            text=text,
            source="asr",
        ))
    logger.success(
        f"[Dialogue] ASR 完成: {len(out)} 段, 检测语言={info.language} "
        f"(prob={info.language_probability:.2f})"
    )
    # 释放显存
    del model
    return out


# ---------------------------------------------------------------------------
# 顶层 Stage
# ---------------------------------------------------------------------------

class DialogueStage:
    """对白获取阶段封装。"""

    def __init__(self, config: dict) -> None:
        self._config = config
        whisper_cfg = config.get("whisper", {})
        self._whisper_model: str = whisper_cfg.get("model_size", "large-v3")
        self._whisper_device: str = whisper_cfg.get("device", "cuda")
        self._whisper_compute: str = whisper_cfg.get(
            "compute_type", "int8_float16"
        )
        self._whisper_language: Optional[str] = whisper_cfg.get("language")
        temp_dir = config.get("paths", {}).get("temp_dir", "/tmp/ai_video_tmp")
        self._work_dir = Path(temp_dir) / "dialogue"
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        video_path: str,
        srt_path: Optional[str] = None,
    ) -> List[DialogueLine]:
        """获取对白：SRT 优先，否则 ASR。

        Args:
            video_path: 输入视频路径。
            srt_path:   外挂 SRT 字幕路径，None 则走 ASR。

        Returns:
            按时间排序的 DialogueLine 列表（可能为空）。
        """
        if srt_path and Path(srt_path).exists():
            try:
                return parse_srt(srt_path)
            except Exception as exc:
                logger.warning(
                    f"[Dialogue] SRT 解析失败 ({exc})，回退到 ASR"
                )

        if not shutil.which("ffmpeg"):
            raise RuntimeError("[Dialogue] 系统未安装 ffmpeg")

        wav_path = str(self._work_dir / "audio_16k.wav")
        extract_audio_track(video_path, wav_path)
        return transcribe_with_whisper(
            wav_path,
            model_size=self._whisper_model,
            device=self._whisper_device,
            compute_type=self._whisper_compute,
            language=self._whisper_language,
        )
