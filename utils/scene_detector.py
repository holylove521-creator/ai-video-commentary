"""
场景变化检测器

基于帧间直方图差异，快速定位视频中的场景切换点，
大幅减少需要送入 VL 模型的帧数（从每秒1帧 → 仅关键帧）。
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


class SceneDetector:
    """基于直方图差异的场景切换检测器。

    Args:
        threshold:      帧间 Bhattacharyya 距离阈值（0-1），越大越不敏感。
                        推荐范围 0.3–0.5，默认 0.4。
        min_scene_gap:  相邻关键帧最小间隔（秒），避免同一场景重复采样。
        histogram_bins: 每通道直方图 bin 数，越小越快但精度略低。
        resize_for_hist: 计算直方图时缩放到的分辨率，仅影响检测速度。
    """

    def __init__(
        self,
        threshold: float = 0.4,
        min_scene_gap: float = 2.0,
        histogram_bins: int = 16,
        resize_for_hist: tuple[int, int] = (640, 360),
    ) -> None:
        self.threshold = threshold
        self.min_scene_gap = min_scene_gap
        self.histogram_bins = histogram_bins
        self.resize_for_hist = resize_for_hist

    def detect_keyframes(
        self,
        video_path: str,
        temp_dir: str = "/tmp/ai_video_frames",
        jpeg_quality: int = 85,
    ) -> list[dict]:
        """检测场景切换，抽取关键帧并保存为 JPEG。

        Args:
            video_path:   视频文件路径。
            temp_dir:     JPEG 保存目录。
            jpeg_quality: JPEG 压缩质量。

        Returns:
            关键帧信息列表，每项::

                {
                    "frame_idx": int,
                    "timestamp": float,
                    "path": str,          # 保存的 JPEG 路径
                    "scene_change_score": float,  # 与上一关键帧的差异分
                }
        """
        video_path = str(video_path)
        out_dir = Path(temp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps: float = cap.get(cv2.CAP_PROP_FPS)
        if not fps:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        min_gap_frames = int(fps * self.min_scene_gap)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

        logger.info(
            f"[SceneDetector] 开始检测: {video_path}  "
            f"时长={duration:.1f}s  fps={fps:.2f}  "
            f"阈值={self.threshold}  最小间隔={self.min_scene_gap}s"
        )

        keyframes: list[dict] = []
        prev_hist: np.ndarray | None = None
        frame_idx = 0
        last_keyframe_idx = -min_gap_frames

        # 强制把第一帧作为关键帧
        first_frame_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 缩小后计算直方图（仅用于检测，不影响保存质量）
            small = cv2.resize(frame, self.resize_for_hist)
            bins = [self.histogram_bins] * 3
            hist = cv2.calcHist(
                [small], [0, 1, 2], None, bins,
                [0, 256, 0, 256, 0, 256],
            )
            cv2.normalize(hist, hist)
            hist = hist.flatten()

            is_keyframe = False
            score = 0.0

            if not first_frame_saved:
                is_keyframe = True
                first_frame_saved = True
            elif prev_hist is not None:
                score = float(cv2.compareHist(
                    prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA
                ))
                gap_ok = (frame_idx - last_keyframe_idx) >= min_gap_frames
                if score > self.threshold and gap_ok:
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
            f"[SceneDetector] 检测完成: {len(keyframes)} 个关键帧  "
            f"（原始 {total_frames} 帧，压缩率 "
            f"{100*(1 - len(keyframes)/max(total_frames,1)):.1f}%）"
        )
        return keyframes
