"""
Shot 切分器

新版基于 **PySceneDetect** ``ContentDetector``，输入为 NVDEC 抽出的 360p
缩略图序列，速度快、精度好。无 PySceneDetect 时回退到自实现的直方图差异。

主入口 :meth:`SceneDetector.detect_shots`，输出 :class:`pipeline.schema.Shot` 列表。

旧的 :meth:`detect_keyframes` 接口保留以兼容旧流水线（仅做关键帧抽取，不再推荐）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from pipeline.schema import Shot


class SceneDetector:
    """Shot 切分器（PySceneDetect 优先，直方图回退）。

    Args:
        threshold:      ContentDetector 阈值（默认 27.0）。直方图模式下
                        作为 Bhattacharyya 阈值（0-1）。
        min_scene_len:  PySceneDetect ``min_scene_len`` 帧数。
        min_scene_gap:  直方图回退路径下的最短场景间隔（秒）。
        histogram_bins: 直方图回退路径下每通道 bin 数。
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 12,
        min_scene_gap: float = 0.8,
        histogram_bins: int = 16,
    ) -> None:
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.min_scene_gap = min_scene_gap
        self.histogram_bins = histogram_bins

    # ------------------------------------------------------------------
    # 新接口：基于缩略图序列的 shot 切分
    # ------------------------------------------------------------------

    def detect_shots(
        self,
        thumbs: list[dict],
        thumbs_fps: float = 2.0,
    ) -> list[Shot]:
        """基于缩略图序列切分 shot。

        Args:
            thumbs:     :meth:`FrameExtractor.extract_thumbnails_nvdec` 输出的列表
                        ``[{frame_idx, timestamp, path}]``。
            thumbs_fps: 缩略图序列的实际 fps（必须与抽帧时一致）。

        Returns:
            :class:`Shot` 列表，``start/end`` 单位秒，覆盖整段视频。
        """
        if not thumbs:
            return []

        # 优先使用 PySceneDetect（基于缩略图的"伪视频"或直接读图序列）
        try:
            return self._detect_with_pyscenedetect(thumbs, thumbs_fps)
        except ImportError:
            logger.warning(
                "[SceneDetector] PySceneDetect 未安装，回退直方图差异。"
                "建议: pip install scenedetect"
            )
        except Exception as exc:
            logger.warning(
                f"[SceneDetector] PySceneDetect 失败 ({exc})，回退直方图"
            )

        return self._detect_with_histogram(thumbs, thumbs_fps)

    # ------------------------------------------------------------------
    # PySceneDetect 实现（在缩略图序列上跑 ContentDetector）
    # ------------------------------------------------------------------

    def _detect_with_pyscenedetect(
        self,
        thumbs: list[dict],
        thumbs_fps: float,
    ) -> list[Shot]:
        from scenedetect import ContentDetector  # type: ignore

        # 自实现"在图像序列上跑 ContentDetector"，避免必须传入 video file
        detector = ContentDetector(
            threshold=float(self.threshold),
            min_scene_len=int(self.min_scene_len),
        )
        cuts: list[int] = []  # 0-based thumb 序号
        for idx, t in enumerate(thumbs):
            img = cv2.imread(t["path"])
            if img is None:
                continue
            # ContentDetector.process_frame 接受 BGR ndarray
            try:
                ev = detector.process_frame(idx, img)
            except TypeError:
                # 老版本签名不一致，回退到直方图
                raise
            if ev:
                # 兼容不同版本：可能返回 list[int] 或 list[FrameTimecode]
                for c in ev:
                    cuts.append(int(c) if isinstance(c, int) else int(getattr(c, "frame_num", 0)))

        cuts = sorted(set(cuts))
        return self._cuts_to_shots(cuts, thumbs, thumbs_fps)

    # ------------------------------------------------------------------
    # 回退：直方图差异（不依赖 PySceneDetect）
    # ------------------------------------------------------------------

    def _detect_with_histogram(
        self,
        thumbs: list[dict],
        thumbs_fps: float,
    ) -> list[Shot]:
        period = 1.0 / max(thumbs_fps, 1e-6)
        min_gap_thumbs = max(1, int(round(self.min_scene_gap / period)))

        prev_hist: Optional[np.ndarray] = None
        cuts: list[int] = []
        last_cut = -min_gap_thumbs
        # 直方图阈值（Bhattacharyya）：把 ContentDetector 的 0-100 映射回 0-1
        bhat_thr = self.threshold / 100.0 if self.threshold > 1 else self.threshold

        for idx, t in enumerate(thumbs):
            img = cv2.imread(t["path"])
            if img is None:
                continue
            bins = [self.histogram_bins] * 3
            hist = cv2.calcHist(
                [img], [0, 1, 2], None, bins,
                [0, 256, 0, 256, 0, 256],
            )
            cv2.normalize(hist, hist)
            hist = hist.flatten()
            if prev_hist is not None:
                score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if score > bhat_thr and (idx - last_cut) >= min_gap_thumbs:
                    cuts.append(idx)
                    last_cut = idx
            prev_hist = hist

        return self._cuts_to_shots(cuts, thumbs, thumbs_fps)

    # ------------------------------------------------------------------
    # 公共：cuts → Shot
    # ------------------------------------------------------------------

    @staticmethod
    def _cuts_to_shots(
        cuts: list[int],
        thumbs: list[dict],
        thumbs_fps: float,
    ) -> list[Shot]:
        """把 cut 点列表（thumb idx）转成完整 Shot 列表。"""
        # 用每个 thumb 的 timestamp 做精确时间映射
        timestamps = [float(t["timestamp"]) for t in thumbs]
        if not timestamps:
            return []
        full_start = max(0.0, timestamps[0] - 0.5 / thumbs_fps)
        full_end = timestamps[-1] + 0.5 / thumbs_fps

        # 把 cuts 翻成 (start_idx, end_idx) 区间
        boundary_idxs = [0] + sorted(set(cuts)) + [len(thumbs)]
        shots: list[Shot] = []
        for sid, (a, b) in enumerate(zip(boundary_idxs, boundary_idxs[1:])):
            if a >= b:
                continue
            start_ts = timestamps[a] - 0.5 / thumbs_fps if a < len(timestamps) else full_start
            end_ts = timestamps[b - 1] + 0.5 / thumbs_fps
            start_ts = max(0.0, start_ts)
            # 代表帧取中间那张
            mid = (a + b - 1) // 2
            shots.append(Shot(
                shot_id=sid,
                start=round(start_ts, 3),
                end=round(end_ts, 3),
                repr_frame_path=thumbs[mid]["path"],
            ))

        # 防御：把首尾对齐到完整视频
        if shots:
            shots[0].start = round(min(shots[0].start, full_start), 3)
            shots[-1].end = round(max(shots[-1].end, full_end), 3)

        logger.info(
            f"[SceneDetector] shot 切分: {len(shots)} 个 "
            f"(cuts={len(cuts)}, thumbs={len(thumbs)})"
        )
        return shots

    # ------------------------------------------------------------------
    # Legacy: 关键帧抽取（保留以避免破坏旧流水线）
    # ------------------------------------------------------------------

    def detect_keyframes(
        self,
        video_path: str,
        temp_dir: str = "/tmp/ai_video_frames",
        jpeg_quality: int = 85,
    ) -> list[dict]:
        """旧接口：在原视频上做直方图差异关键帧抽取。

        新代码请使用 :meth:`detect_shots`。
        """
        video_path = str(video_path)
        out_dir = Path(temp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        min_gap_frames = int(fps * self.min_scene_gap)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        bhat_thr = self.threshold / 100.0 if self.threshold > 1 else self.threshold

        keyframes: list[dict] = []
        prev_hist: Optional[np.ndarray] = None
        frame_idx = 0
        last_keyframe_idx = -min_gap_frames
        first_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (640, 360))
            bins = [self.histogram_bins] * 3
            hist = cv2.calcHist([small], [0, 1, 2], None, bins,
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            hist = hist.flatten()

            is_keyframe = False
            score = 0.0
            if not first_saved:
                is_keyframe = True
                first_saved = True
            elif prev_hist is not None:
                score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                gap_ok = (frame_idx - last_keyframe_idx) >= min_gap_frames
                if score > bhat_thr and gap_ok:
                    is_keyframe = True

            if is_keyframe:
                out_path = out_dir / f"key_{frame_idx:08d}.jpg"
                cv2.imwrite(str(out_path), frame, encode_params)
                keyframes.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                    "path": str(out_path),
                    "scene_change_score": round(score, 4),
                })
                last_keyframe_idx = frame_idx
            prev_hist = hist
            frame_idx += 1

        cap.release()
        logger.info(
            f"[SceneDetector] (legacy) 关键帧 {len(keyframes)} / 原始 "
            f"{total_frames} 帧"
        )
        return keyframes
