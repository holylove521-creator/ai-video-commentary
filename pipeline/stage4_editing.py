"""
Stage 4: 智能剪辑与视频合成（纯 ffmpeg 流水线）

设计目标：4K/2h 输入下尽量避免重新编码视频流。

策略：
1. 为每个 ``MixSegment`` 在源视频上选择一个时长等于 TTS 实际音频时长的素材区间
   （以该段绑定 scene 的中点为锚），用 ``ffmpeg -c copy`` 流拷贝切片成 ``clip_NNNN.mp4``。
2. 用 concat demuxer 将所有 clip 拼成一条无音轨的 ``concat.mp4``（继续 ``-c copy``）。
3. 将 TTS 音频按 MixSegment 顺序拼接（含章节静音），生成 ``narration_full.wav``。
4. 由 ``SubtitleRenderer`` 生成 ASS 字幕（按 MixSegment 输出时间轴）。
5. 最后一次 ``ffmpeg`` 调用合并视频 + 音频 + 字幕烧录，使用 ``h264_nvenc`` 一次性编码。

不再依赖 MoviePy / variable-speed 调整。
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from utils.subtitle_renderer import SubtitleRenderer
from pipeline.schema import MixSegment, Scene


class VideoEditor:
    """基于 ffmpeg 的视频剪辑与合成器。

    Args:
        config: 全局配置字典。
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        video_cfg = config.get("video", {})
        self._output_codec: str = video_cfg.get("output_codec", "h264_nvenc")
        self._output_cq: int = int(video_cfg.get("output_cq", 23))
        self._output_preset: str = video_cfg.get("output_preset", "p4")
        self._audio_bitrate: str = video_cfg.get("audio_bitrate", "192k")
        self._use_cuda_decode: bool = bool(video_cfg.get("use_cuda_decode", True))

        temp_dir = config.get("paths", {}).get("temp_dir", "/tmp/ai_video_tmp")
        self._work_dir = Path(temp_dir) / "editing"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._clip_dir = self._work_dir / "clips"
        self._clip_dir.mkdir(parents=True, exist_ok=True)

        # 字幕样式（从 style_templates 读取，若不可用则回退默认）
        sub_style = (config.get("subtitle_style") or {})
        self._subtitle_renderer = SubtitleRenderer(
            font_name=sub_style.get("font_name", "微软雅黑"),
            font_size=int(sub_style.get("font_size", 42)),
            primary_color=sub_style.get("primary_color", "&H00FFFFFF"),
            outline_color=sub_style.get("outline_color", "&H00000000"),
        )

    # ------------------------------------------------------------------
    # 主合成入口
    # ------------------------------------------------------------------

    def compose(
        self,
        video_path: str,
        mix_segments: list[MixSegment],
        output_path: str,
        scenes: Optional[list[Scene]] = None,
        no_subtitle: bool = False,
    ) -> str:
        """合成最终视频。

        Args:
            video_path:    源视频路径。
            mix_segments:  Stage3 输出，``start_time/end_time`` 已是输出时间轴。
            output_path:   目标输出 mp4 路径。
            scenes:        Stage1 产出的 Scene 列表，用于查找每段对应的源时间区间。
            no_subtitle:   是否跳过字幕烧录。
        """
        if not mix_segments:
            raise RuntimeError("[Stage4] mix_segments 为空，无法合成")

        source_duration = self._probe_duration(video_path)
        scene_index = self._build_scene_index(scenes or [])

        # 1) 为每段选择源区间并切片
        clip_files = self._cut_clips(video_path, mix_segments, source_duration, scene_index)

        # 2) concat 视频（无音轨）
        concat_video = str(self._work_dir / "concat.mp4")
        self._concat_clips(clip_files, concat_video)

        # 3) 拼接 TTS 音频
        narration_wav = str(self._work_dir / "narration_full.wav")
        self._concat_narration(mix_segments, narration_wav)

        # 4) 生成 ASS 字幕
        ass_path: Optional[str] = None
        if not no_subtitle:
            ass_path = self._generate_ass(mix_segments)

        # 5) 最终一次性编码
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._final_encode(concat_video, narration_wav, ass_path, output_path)

        logger.success(f"[Stage4] 合成完成 → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 切片
    # ------------------------------------------------------------------

    def _cut_clips(
        self,
        video_path: str,
        mix_segments: list[MixSegment],
        source_duration: float,
        scene_index: dict[int, tuple[float, float]],
    ) -> list[str]:
        """根据每个 MixSegment 的实际音频时长，从源视频上切出等长片段。"""
        out: list[str] = []
        for i, seg in enumerate(tqdm(mix_segments, desc="切片")):
            actual_dur = float(
                (seg.extra or {}).get("actual_duration",
                                      max(0.5, seg.end_time - seg.start_time))
            )
            actual_dur = max(0.5, actual_dur)

            src_in = self._pick_source_window(
                seg, actual_dur, source_duration, scene_index
            )
            src_out = min(source_duration, src_in + actual_dur)

            clip_path = str(self._clip_dir / f"clip_{i:04d}.mp4")
            self._stream_copy_cut(video_path, src_in, src_out, clip_path)
            out.append(clip_path)
        return out

    def _pick_source_window(
        self,
        seg: MixSegment,
        need_dur: float,
        source_duration: float,
        scene_index: dict[int, tuple[float, float]],
    ) -> float:
        """选择源视频上的入点，以匹配 TTS 时长。"""
        narration = (seg.extra or {}).get("narration") or {}
        scene_id = (
            narration.get("source_scene_id")
            if isinstance(narration, dict)
            else None
        )
        if scene_id is None:
            scene_id = (seg.extra or {}).get("source_scene_id")

        if isinstance(scene_id, int) and scene_id in scene_index:
            s_start, s_end = scene_index[scene_id]
            mid = (s_start + s_end) / 2.0
            src_in = mid - need_dur / 2.0
        else:
            # 退化：按输出时间轴比例落到源
            src_in = (seg.start_time / max(1.0, seg.end_time)) * source_duration

        src_in = max(0.0, min(src_in, max(0.0, source_duration - need_dur)))
        return src_in

    def _stream_copy_cut(
        self, video_path: str, t_in: float, t_out: float, out_path: str
    ) -> None:
        """流拷贝切片（无音频）。"""
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t_in:.3f}",
            "-to", f"{t_out:.3f}",
            "-i", video_path,
            "-an",
            "-c:v", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # 关键帧对不齐时回退为 nvenc 重编码该小片段
            logger.warning(
                f"[Stage4] 流拷贝失败，回退 nvenc 编码切片: {Path(out_path).name}"
            )
            cmd2 = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{t_in:.3f}",
                "-to", f"{t_out:.3f}",
                "-i", video_path,
                "-an",
                "-c:v", self._output_codec,
                "-preset", self._output_preset,
                "-cq", str(self._output_cq),
                out_path,
            ]
            subprocess.run(cmd2, check=True)

    # ------------------------------------------------------------------
    # 拼接
    # ------------------------------------------------------------------

    def _concat_clips(self, clip_files: list[str], out_path: str) -> None:
        list_file = self._work_dir / "concat_video.txt"
        list_file.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in clip_files),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.warning("[Stage4] 视频 concat 流拷贝失败，回退重编码")
            cmd2 = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c:v", self._output_codec,
                "-preset", self._output_preset,
                "-cq", str(self._output_cq),
                out_path,
            ]
            subprocess.run(cmd2, check=True)

    def _concat_narration(
        self, mix_segments: list[MixSegment], out_wav: str
    ) -> None:
        """拼接所有 TTS wav；段间静音由 MixSegment 时间轴决定。"""
        # 依据 MixSegment.start_time 间隙生成静音
        items: list[str] = []
        cursor = 0.0
        silence_dir = self._work_dir / "silence"
        silence_dir.mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(mix_segments):
            gap = max(0.0, seg.start_time - cursor)
            if gap > 0.01:
                sil_path = str(silence_dir / f"sil_{i:04d}.wav")
                self._make_silence(sil_path, gap)
                items.append(sil_path)
            if seg.narration_audio and Path(seg.narration_audio).exists():
                items.append(seg.narration_audio)
            cursor = seg.end_time

        if not items:
            raise RuntimeError("[Stage4] 没有可用的 TTS 音频")

        list_file = self._work_dir / "concat_audio.txt"
        list_file.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in items),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-ar", "48000", "-ac", "2",
            out_wav,
        ]
        subprocess.run(cmd, check=True)

    def _make_silence(self, out_path: str, duration: float) -> None:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"anullsrc=r=48000:cl=mono",
            "-t", f"{duration:.3f}",
            out_path,
        ]
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # 字幕
    # ------------------------------------------------------------------

    def _generate_ass(self, mix_segments: list[MixSegment]) -> str:
        ass_path = str(self._work_dir / "subtitles.ass")
        script = []
        for seg in mix_segments:
            narration = (seg.extra or {}).get("narration")
            text = ""
            if isinstance(narration, dict):
                text = narration.get("text", "") or ""
            if not text:
                continue
            script.append({
                "start": float(seg.start_time),
                "end": float(seg.end_time),
                "text": text,
            })
        return self._subtitle_renderer.generate_ass(script, ass_path)

    # ------------------------------------------------------------------
    # 最终编码
    # ------------------------------------------------------------------

    def _final_encode(
        self,
        concat_video: str,
        narration_wav: str,
        ass_path: Optional[str],
        output_path: str,
    ) -> None:
        """合并视频 + 音频；如有字幕则烧录。"""
        cmd = ["ffmpeg", "-y", "-loglevel", "info"]
        if self._use_cuda_decode:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-i", concat_video, "-i", narration_wav]

        if ass_path:
            ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            vf = f"subtitles='{ass_escaped}'"
            cmd += ["-filter_complex", f"[0:v]{vf}[v]",
                    "-map", "[v]", "-map", "1:a"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]

        cmd += [
            "-c:v", self._output_codec,
            "-preset", self._output_preset,
            "-rc", "vbr",
            "-cq", str(self._output_cq),
            "-c:a", "aac",
            "-b:a", self._audio_bitrate,
            "-shortest",
            output_path,
        ]
        logger.info(f"[Stage4] 最终编码: {' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        try:
            return float(out)
        except ValueError:
            return 0.0

    @staticmethod
    def _build_scene_index(
        scenes: list[Scene],
    ) -> dict[int, tuple[float, float]]:
        idx: dict[int, tuple[float, float]] = {}
        for s in scenes:
            try:
                idx[int(s.scene_id)] = (float(s.start), float(s.end))
            except Exception:
                continue
        return idx


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
    sample_script = [
        NarrationSegment(
            text="欢迎观看今天的视频！",
            event_block_index=0, speaker="旁白", style=None,
            start_time=0.0, end_time=5.0,
            extra={"emotion": "excited", "source_scene_id": 0, "chapter_id": 1},
        ),
        NarrationSegment(
            text="精彩内容即将呈现。",
            event_block_index=1, speaker="旁白", style=None,
            start_time=5.0, end_time=10.0,
            extra={"emotion": "calm", "source_scene_id": 1, "chapter_id": 1},
        ),
    ]
    mix_segments = engine.synthesize_all(sample_script)

    editor = VideoEditor(config)
    no_sub = "--no-subtitle" in sys.argv
    result = editor.compose(
        video_path=sys.argv[1],
        mix_segments=mix_segments,
        output_path=sys.argv[2],
        scenes=None,
        no_subtitle=no_sub,
    )
    print(f"输出视频: {result}")
