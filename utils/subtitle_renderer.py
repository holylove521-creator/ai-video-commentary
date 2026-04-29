"""
字幕生成与烧录工具

将结构化脚本转换为 ASS 格式字幕文件，
并通过 FFmpeg（h264_nvenc 硬件编码）将字幕烧录到视频中。
"""

import subprocess
from pathlib import Path

from loguru import logger


class SubtitleRenderer:
    """ASS 字幕生成与视频烧录工具。

    支持自定义字体、字号、颜色，兼容中文字体，
    并使用 NVIDIA h264_nvenc 硬件加速输出。
    """

    # ASS 文件头模板
    _ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def __init__(
        self,
        font_name: str = "微软雅黑",
        font_size: int = 48,
        primary_color: str = "&H00FFFFFF",
        outline_color: str = "&H00000000",
    ) -> None:
        """初始化字幕渲染器。

        Args:
            font_name:     字体名称，默认微软雅黑。
            font_size:     字号（像素），默认 48。
            primary_color: 字体颜色（ASS ABGR 格式），默认白色。
            outline_color: 描边颜色，默认黑色。
        """
        self.font_name = font_name
        self.font_size = font_size
        self.primary_color = primary_color
        self.outline_color = outline_color

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_ass(self, script: list[dict], output_path: str) -> str:
        """将脚本列表生成 ASS 格式字幕文件。

        Args:
            script: 脚本片段列表，每项格式::

                {"start": float, "end": float, "text": str}

            output_path: 输出 ASS 文件路径。

        Returns:
            实际写入的 ASS 文件路径字符串。
        """
        header = self._ASS_HEADER.format(
            font_name=self.font_name,
            font_size=self.font_size,
            primary_color=self.primary_color,
            outline_color=self.outline_color,
        )

        lines: list[str] = [header]
        for seg in script:
            start_str = self._seconds_to_ass_time(float(seg["start"]))
            end_str = self._seconds_to_ass_time(float(seg["end"]))
            text = str(seg.get("text", "")).replace("\n", "\\N")
            lines.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}"
            )

        content = "\n".join(lines) + "\n"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8-sig")
        logger.info(
            f"[SubtitleRenderer] 生成字幕文件: {out_path}  ({len(script)} 段)"
        )
        return str(out_path)

    def burn_to_video(
        self,
        video_path: str,
        ass_path: str,
        output_path: str,
    ) -> None:
        """使用 FFmpeg 将 ASS 字幕烧录到视频，启用 h264_nvenc 硬件编码。

        Args:
            video_path:  输入视频路径。
            ass_path:    ASS 字幕文件路径。
            output_path: 输出视频路径。

        Raises:
            subprocess.CalledProcessError: FFmpeg 命令执行失败。
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 转义冒号与反斜杠，避免 ffmpeg 滤镜语法冲突
        safe_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass={safe_ass}",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-c:a", "copy",
            output_path,
        ]
        logger.info(f"[SubtitleRenderer] 烧录字幕: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        logger.success(f"[SubtitleRenderer] 字幕烧录完成 → {output_path}")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """将秒数转换为 ASS 时间格式 ``H:MM:SS.cc``。

        Args:
            seconds: 秒数（可含小数）。

        Returns:
            ASS 格式时间字符串，例如 ``0:01:23.45``。
        """
        total_cs = int(round(seconds * 100))
        cs = total_cs % 100
        total_s = total_cs // 100
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
