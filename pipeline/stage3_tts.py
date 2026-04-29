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

    def synthesize_all(self, script: list[dict]) -> list[dict]:
        """批量合成全部脚本段落。

        Args:
            script: Stage 2 输出的脚本列表。

        Returns:
            在原脚本基础上增加 ``audio_path`` 字段的列表。
        """
        results: list[dict] = []
        for i, seg in enumerate(tqdm(script, desc="语音合成")):
            audio_path = self.synthesize_segment(
                text=seg.get("text", ""),
                emotion=seg.get("emotion", "neutral"),
                segment_idx=i,
            )
            results.append({**seg, "audio_path": audio_path})
        logger.success(f"[Stage3] 全部 {len(results)} 段语音合成完成")
        return results

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

    sample_script = [
        {"start": 0.0, "end": 5.0, "text": "欢迎来到今天的游戏直播！", "emotion": "excited"},
        {"start": 5.0, "end": 10.0, "text": "我们即将迎来精彩的对决。", "emotion": "tense"},
        {"start": 10.0, "end": 15.0, "text": "这波操作真的太秀了！", "emotion": "funny"},
    ]

    results = engine.synthesize_all(sample_script)
    for seg in results:
        dur = engine.get_audio_duration(seg["audio_path"])
        print(
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] "
            f"音频: {seg['audio_path']}  时长: {dur:.2f}s"
        )
