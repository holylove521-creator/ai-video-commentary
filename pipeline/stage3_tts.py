"""
Stage 3: 语音合成（TTS）

使用 CosyVoice2-0.5B 将解说脚本逐段合成语音，
支持情感标签映射和声音克隆，输出 WAV 格式音频文件。

CosyVoice2 未安装时自动降级：生成静音 WAV 占位文件，并输出安装提示。

输出目录: /tmp/ai_video_tts/
"""

import asyncio
import os
import struct
import wave
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm
from pipeline.schema import NarrationSegment, MixSegment


# 情感标签 → CosyVoice2 instructed speech 前缀映射
EMOTION_PROMPT_MAP: dict[str, str] = {
    "excited": "用兴奋激动的语气说：",
    "calm":    "用平静自然的语气说：",
    "serious": "用严肃深沉的语气说：",
    "funny":   "用幽默搞笑的语气说：",
    "tense":   "用紧张急促的语气说：",
    "neutral": "",
}


class TTSEngine:
    """CosyVoice2 语音合成引擎（延迟加载，避免与 llama.cpp 争抢显存）。

    Args:
        config:     全局配置字典。
        ref_audio:  声音克隆参考音频路径（可选，3-10 秒干净人声 WAV）。
    """

    def __init__(
        self,
        config: dict,
        ref_audio: Optional[str] = None,
    ) -> None:
        self._config = config
        tts_cfg = config.get("tts", {})
        self._model_path: str = tts_cfg.get("model_path", "models/CosyVoice2-0.5B")
        self._default_voice: str = tts_cfg.get("default_voice", "zh-cn-female-1")
        self._sample_rate: int = int(tts_cfg.get("sample_rate", 22050))
        self._max_concurrent: int = int(tts_cfg.get("max_concurrent", 4))
        self._max_chars_per_sentence: int = int(tts_cfg.get("max_chars_per_sentence", 30))
        self._chapter_gap_seconds: float = float(tts_cfg.get("chapter_gap_seconds", 0.3))
        self._ref_audio: Optional[str] = (
            self._resolve_safe_path(ref_audio) if ref_audio else None
        )

        self._model = None          # 延迟加载
        self._cosyvoice_available = False
        self._output_dir = Path(
            config.get("paths", {}).get("temp_dir", "/tmp/ai_video_tmp")
        ) / "tts"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 模型加载（延迟）
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> bool:
        """懒加载 CosyVoice2 模型。

        Returns:
            ``True`` 表示加载成功，``False`` 表示 CosyVoice2 未安装（降级模式）。
        """
        if self._model is not None:
            return self._cosyvoice_available

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
            logger.info(f"[Stage3] 加载 CosyVoice2 模型: {self._model_path}")
            self._model = CosyVoice2(self._model_path)
            self._cosyvoice_available = True
            logger.success("[Stage3] CosyVoice2 模型加载成功")
        except ImportError:
            logger.warning(
                "[Stage3] CosyVoice2 未安装，进入降级模式（生成静音占位音频）。\n"
                "安装方法：\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice\n"
                "  cd CosyVoice && pip install -e ."
            )
            self._cosyvoice_available = False

        return self._cosyvoice_available
    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def synthesize_segment(
        self,
        text: str,
        emotion: str = "neutral",
        segment_idx: int = 0,
    ) -> str:
        """合成单段语音。

        Args:
            text:         解说文本。
            emotion:      情感标签（映射到 CosyVoice2 instructed prompt）。
            segment_idx:  段落序号，用于生成唯一文件名。

        Returns:
            输出 WAV 文件路径。
        """
        out_path = str(self._output_dir / f"seg_{segment_idx:04d}.wav")

        if not self._ensure_model_loaded():
            self._write_silence(out_path, duration=3.0)
            return out_path

        instruction = EMOTION_PROMPT_MAP.get(emotion, "")
        tts_text = f"{instruction}{text}" if instruction else text

        try:
            import torchaudio  # type: ignore
            if self._ref_audio is not None:
                # 声音克隆模式
                result = self._model.inference_zero_shot(
                    tts_text=tts_text,
                    prompt_speech_16k=self._load_ref_audio(),
                    stream=False,
                )
            else:
                # 预置声音模式
                result = self._model.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text=instruction or "用自然的语气说：",
                    spk_id=self._default_voice,
                    stream=False,
                )
            saved = False
            for batch in result:
                torchaudio.save(
                    out_path,
                    batch["tts_speech"],
                    self._model.sample_rate,
                )
                saved = True
                break  # 取第一批输出
            if not saved:
                logger.warning(
                    f"[Stage3] 段落 {segment_idx} TTS 生成器返回空结果，生成静音"
                )
                self._write_silence(out_path, duration=3.0)
        except Exception as exc:
            logger.error(f"[Stage3] 段落 {segment_idx} 合成失败: {exc}，生成静音")
            self._write_silence(out_path, duration=3.0)

        return out_path

    def synthesize_all(self, narration_segments: list[NarrationSegment]) -> list[MixSegment]:
        """批量合成 NarrationSegment → MixSegment。

        新策略：
        - 单段文本超过 ``max_chars_per_sentence`` 时按标点切分后合成再拼接（避免
          CosyVoice2 长句吞字）。
        - **以 TTS 实际生成的音频时长为准**，重新计算 ``start_time/end_time``，
          上层剪辑直接以此对齐，不再变速画面。
        - 跨章节边界（``extra.chapter_id`` 变化）插入 ``chapter_gap_seconds`` 静音。
        """
        results: list[MixSegment] = []
        cursor = 0.0
        prev_chapter: Optional[int] = None
        for i, seg in enumerate(tqdm(narration_segments, desc="语音合成")):
            emotion = (seg.extra or {}).get("emotion", "neutral")
            chapter_id = (seg.extra or {}).get("chapter_id")

            # 章节边界静音
            if (
                prev_chapter is not None
                and chapter_id is not None
                and chapter_id != prev_chapter
                and self._chapter_gap_seconds > 0
            ):
                cursor += self._chapter_gap_seconds
            prev_chapter = chapter_id

            audio_path = self._synthesize_long_text(
                text=seg.text,
                emotion=emotion,
                segment_idx=i,
            )
            actual_dur = self.get_audio_duration(audio_path)
            if actual_dur <= 0:
                actual_dur = max(1.0, len(seg.text) / 4.0)

            start = round(cursor, 3)
            end = round(cursor + actual_dur, 3)
            cursor = end

            results.append(MixSegment(
                start_time=start,
                end_time=end,
                narration_audio=audio_path,
                subtitle_file=None,
                video_file=None,
                instructions=None,
                extra={
                    "narration": seg.to_dict(),
                    "actual_duration": actual_dur,
                    "chapter_id": chapter_id,
                },
            ))
        logger.success(
            f"[Stage3] 全部 {len(results)} 段语音合成完成，总时长 {cursor:.1f}s"
        )
        return results

    def _synthesize_long_text(
        self,
        text: str,
        emotion: str,
        segment_idx: int,
    ) -> str:
        """长文本按标点切分逐句合成，再用 ffmpeg concat 拼一段 wav。"""
        text = (text or "").strip()
        if not text:
            out_path = str(self._output_dir / f"seg_{segment_idx:04d}.wav")
            self._write_silence(out_path, duration=0.5)
            return out_path

        sentences = self._split_sentences(text, self._max_chars_per_sentence)
        if len(sentences) == 1:
            return self.synthesize_segment(text, emotion, segment_idx)

        sub_paths: list[str] = []
        for j, sent in enumerate(sentences):
            sub_idx = segment_idx * 1000 + j
            sub_paths.append(self.synthesize_segment(sent, emotion, sub_idx))

        out_path = str(self._output_dir / f"seg_{segment_idx:04d}.wav")
        self._concat_wavs(sub_paths, out_path)
        return out_path

    @staticmethod
    def _split_sentences(text: str, max_len: int) -> list[str]:
        """按中文标点切句，超长再按 max_len 硬切。"""
        import re as _re
        # 在标点后切，但保留标点
        parts = _re.split(r"(?<=[。！？!?；;])", text)
        out: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            while len(p) > max_len:
                # 在 max_len 范围内尽量在标点处切
                cut = max_len
                for sep in "，,、 ":
                    pos = p.rfind(sep, 0, max_len)
                    if pos >= max_len // 2:
                        cut = pos + 1
                        break
                out.append(p[:cut].strip())
                p = p[cut:].strip()
            if p:
                out.append(p)
        return out or [text]

    def _concat_wavs(self, wav_paths: list[str], out_path: str) -> None:
        """用 ffmpeg concat demuxer 拼接 wav。"""
        import subprocess
        list_file = Path(out_path).with_suffix(".list.txt")
        list_file.write_text(
            "\n".join(f"file '{Path(p).resolve()}'" for p in wav_paths),
            encoding="utf-8",
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy", out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # 编码不一致时强制重新编码
            cmd2 = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-ar", str(self._sample_rate), "-ac", "1",
                out_path,
            ]
            subprocess.run(cmd2, check=True)
        finally:
            try:
                list_file.unlink(missing_ok=True)
            except Exception:
                pass

    def get_audio_duration(self, audio_path: str) -> float:
        """获取 WAV 音频时长（秒）。

        Args:
            audio_path: WAV 文件路径。

        Returns:
            时长（秒），读取失败时返回 0.0。
        """
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / rate if rate > 0 else 0.0
        except (wave.Error, OSError) as exc:
            logger.warning(f"[Stage3] 读取音频时长失败: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _write_silence(self, path: str, duration: float = 3.0) -> None:
        """写入静音 WAV 文件（降级占位用）。"""
        n_frames = int(self._sample_rate * duration)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(b"\x00\x00" * n_frames)

    @staticmethod
    def _resolve_safe_path(path: str) -> str:
        """解析并验证文件路径，防止路径注入。

        确保路径为已存在的普通文件，并返回其绝对路径。

        Args:
            path: 用户提供的文件路径字符串。

        Returns:
            解析后的绝对路径字符串。

        Raises:
            ValueError: 路径不存在或不是普通文件。
        """
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise ValueError(
                f"参考音频路径无效或不是普通文件: {path}"
            )
        return str(resolved)

    def _load_ref_audio(self):
        """加载并重采样参考音频为 16kHz，供 CosyVoice2 zero-shot 使用。"""
        import torchaudio  # type: ignore
        # self._ref_audio 已在 __init__ 中经过 _resolve_safe_path 验证
        waveform, sr = torchaudio.load(self._ref_audio)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        return waveform


# ------------------------------------------------------------------
# 独立测试入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import yaml

    with open("config/model_config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ref = sys.argv[1] if len(sys.argv) > 1 else None
    engine = TTSEngine(config, ref_audio=ref)

    from pipeline.schema import NarrationSegment
    sample_script = [
        NarrationSegment(text="欢迎来到今天的游戏直播！", event_block_index=0, speaker="旁白", style=None, start_time=0.0, end_time=5.0, extra={"emotion": "excited"}),
        NarrationSegment(text="我们即将迎来精彩的对决。", event_block_index=1, speaker="旁白", style=None, start_time=5.0, end_time=10.0, extra={"emotion": "tense"}),
        NarrationSegment(text="这波操作真的太秀了！", event_block_index=2, speaker="旁白", style=None, start_time=10.0, end_time=15.0, extra={"emotion": "funny"}),
    ]

    results = engine.synthesize_all(sample_script)
    for seg in results:
        dur = engine.get_audio_duration(seg.narration_audio)
        print(
            f"[{seg.start_time:.1f}s - {seg.end_time:.1f}s] "
            f"音频: {seg.narration_audio}  时长: {dur:.2f}s"
        )
