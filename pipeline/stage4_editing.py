"""
Stage 4: 智能剪辑与视频合成

使用 MoviePy 截取视频片段、加载 TTS 音频，
将音频变速对齐画面，添加淡入淡出转场，
最终通过 FFmpeg h264_nvenc 硬件编码输出成品视频。
同时调用 SubtitleRenderer 生成并烧录 ASS 字幕。
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from utils.subtitle_renderer import SubtitleRenderer
from pipeline.schema import MixSegment


class VideoEditor:
    """视频智能剪辑与合成器。

    Args:
        config: 全局配置字典。
    """

    # 允许的变速范围
    SPEED_MIN = 0.7
    SPEED_MAX = 1.5

    def __init__(self, config: dict) -> None:
        self._config = config
        video_cfg = config.get("video", {})
        self._output_codec: str = video_cfg.get("output_codec", "h264_nvenc")
        self._output_crf: int = int(video_cfg.get("output_crf", 23))
        temp_dir = config.get("paths", {}).get("temp_dir", "/tmp/ai_video_tmp")
        self._work_dir = Path(temp_dir) / "editing"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._subtitle_renderer = SubtitleRenderer()

    # ------------------------------------------------------------------
    # 主合成入口
    # ------------------------------------------------------------------

    def compose(
        self,
        video_path: str,
        mix_segments: list[MixSegment],
        output_path: str,
        no_subtitle: bool = False,
    ) -> str:
        """将视频片段与 TTS 音频合成最终视频（输入为 MixSegment）。"""
        try:
            from moviepy.editor import (  # type: ignore
                VideoFileClip,
                AudioFileClip,
                concatenate_videoclips,
            )
        except ImportError as exc:
            raise RuntimeError(
                "moviepy 未安装，请运行 pip install moviepy"
            ) from exc

        logger.info(f"[Stage4] 开始剪辑合成: {video_path}")
        source = VideoFileClip(video_path)
        clips = []

        for i, seg in enumerate(tqdm(mix_segments, desc="剪辑合成")):
            start = float(seg.start_time)
            end = float(seg.end_time)
            end = min(end, source.duration)
            if end <= start:
                logger.warning(f"[Stage4] 段落 {i} 时间无效，跳过")
                continue

            # 截取视频片段
            clip = source.subclip(start, end)

            # 加载 TTS 音频
            audio_path = seg.narration_audio or ""
            if audio_path and Path(audio_path).exists():
                audio = AudioFileClip(audio_path)
                # 计算变速比，将视频片段与音频对齐
                video_dur = clip.duration
                audio_dur = audio.duration
                if audio_dur > 0 and video_dur > 0:
                    speed = video_dur / audio_dur
                    speed = max(self.SPEED_MIN, min(self.SPEED_MAX, speed))
                    clip = clip.speedx(speed)
                clip = clip.set_audio(audio.set_duration(clip.duration))
            else:
                logger.debug(f"[Stage4] 段落 {i} 无音频，保留原音频")

            # 淡入淡出转场（0.3 秒）
            fade_dur = min(0.3, clip.duration / 4)
            clip = clip.fadein(fade_dur).fadeout(fade_dur)
            clips.append(clip)

        if not clips:
            source.close()
            raise RuntimeError("[Stage4] 没有有效片段，合成失败")

        # 拼接所有片段
        final = concatenate_videoclips(clips, method="compose")
        source.close()

        # 使用 FFmpeg 硬件编码写出（moviepy write_videofile 支持 ffmpeg_params）
        temp_output = str(self._work_dir / "temp_composed.mp4")
        logger.info(f"[Stage4] 编码输出（{self._output_codec}）→ {temp_output}")
        final.write_videofile(
            temp_output,
            codec=self._output_codec,
            audio_codec="aac",
            ffmpeg_params=["-preset", "p4", "-cq", str(self._output_crf)],
            logger=None,
        )
        final.close()

        # 字幕处理
        if not no_subtitle:
            output_path = self._attach_subtitles(
                mix_segments, temp_output, output_path
            )
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_output, output_path)

        logger.success(f"[Stage4] 合成完成 → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 字幕
    # ------------------------------------------------------------------

    def generate_subtitles(
        self, mix_segments: list[MixSegment], output_ass: Optional[str] = None
    ) -> str:
        """根据 MixSegment 生成 ASS 字幕文件。"""
        if output_ass is None:
            output_ass = str(self._work_dir / "subtitles.ass")
        # 提取字幕信息
        script = []
        for seg in mix_segments:
            narration = seg.extra.get("narration") if seg.extra else None
            text = narration["text"] if isinstance(narration, dict) else ""
            script.append({
                "start": seg.start_time,
                "end": seg.end_time,
                "text": text,
            })
        return self._subtitle_renderer.generate_ass(script, output_ass)

    def burn_subtitles(
        self, video_path: str, ass_path: str, output_path: str
    ) -> str:
        """将 ASS 字幕烧录到视频。

        Args:
            video_path:  输入视频路径。
            ass_path:    ASS 字幕文件路径。
            output_path: 输出视频路径。

        Returns:
            输出视频路径。
        """
        self._subtitle_renderer.burn_to_video(video_path, ass_path, output_path)
        return output_path

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _attach_subtitles(
        self,
        mix_segments: list[MixSegment],
        video_path: str,
        output_path: str,
    ) -> str:
        """生成字幕并烧录到视频，返回最终输出路径。"""
        try:
            ass_path = self.generate_subtitles(mix_segments)
            self.burn_subtitles(video_path, ass_path, output_path)
        except subprocess.CalledProcessError as exc:
            logger.error(
                f"[Stage4] 字幕烧录失败（{exc}），尝试不带字幕输出"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            os.replace(video_path, output_path)
        return output_path


# ------------------------------------------------------------------
# 独立测试入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    if len(sys.argv) < 3:
        print(
            "用法: python -m pipeline.stage4_editing "
            "<input_video> <output_video> [--no-subtitle]"
        )
        sys.exit(1)

    with open("config/model_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from pipeline.stage3_tts import TTSEngine
    from pipeline.schema import NarrationSegment
    engine = TTSEngine(config)

    # 示例脚本（NarrationSegment）
    sample_script = [
        NarrationSegment(text="欢迎观看今天的视频！", event_block_index=0, speaker="旁白", style=None, start_time=0.0, end_time=5.0, extra={"emotion": "excited"}),
        NarrationSegment(text="精彩内容即将呈现。", event_block_index=1, speaker="旁白", style=None, start_time=5.0, end_time=10.0, extra={"emotion": "calm"}),
    ]
    mix_segments = engine.synthesize_all(sample_script)

    editor = VideoEditor(config)
    no_sub = "--no-subtitle" in sys.argv
    result = editor.compose(
        video_path=sys.argv[1],
        mix_segments=mix_segments,
        output_path=sys.argv[2],
        no_subtitle=no_sub,
    )
    print(f"输出视频: {result}")
