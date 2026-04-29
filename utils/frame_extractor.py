"""
视频抽帧工具，基于 OpenCV

从视频文件中按指定频率抽取帧，保存为 JPEG 图片，
并返回带时间戳的帧信息列表，供后续 VL 模型分析使用。
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger


class FrameExtractor:
    """基于 OpenCV 的视频抽帧工具。

    将视频按指定频率抽帧并保存到临时目录；处理完毕后可调用
    :meth:`cleanup` 删除临时文件以释放磁盘空间。
    """

    def __init__(self, temp_dir: str = "/tmp/ai_video_frames") -> None:
        """初始化抽帧工具。

        Args:
            temp_dir: 帧图片的临时存放目录，若不存在则自动创建。
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[FrameExtractor] 临时目录: {self.temp_dir}")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract(
        self,
        video_path: str,
        fps_sample: float = 1.0,
        quality: int = 85,
    ) -> list[dict]:
        """从视频抽取帧。

        Args:
            video_path: 视频文件路径。
            fps_sample: 抽帧频率（帧/秒），例如 1.0 表示每秒抽 1 帧。
            quality:    JPEG 压缩质量（0-100），默认 85。

        Returns:
            帧信息列表，每项为::

                {
                    "frame_idx": int,    # 原始帧序号
                    "timestamp": float,  # 时间戳（秒）
                    "path": str,         # 保存的 JPEG 路径
                }

        Raises:
            FileNotFoundError: 视频文件不存在。
            RuntimeError:      OpenCV 无法打开视频。
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

        try:
            video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps

            # 每隔多少帧抽一帧
            frame_interval = max(1, int(round(video_fps / fps_sample)))
            logger.info(
                f"[FrameExtractor] 视频: {video_path}  "
                f"时长: {duration:.1f}s  原始 FPS: {video_fps:.2f}  "
                f"抽帧间隔: {frame_interval} 帧"
            )

            frames: list[dict] = []
            frame_idx = 0
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / video_fps
                    out_path = self.temp_dir / f"frame_{frame_idx:08d}.jpg"
                    cv2.imwrite(str(out_path), frame, encode_params)
                    frames.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp": round(timestamp, 3),
                            "path": str(out_path),
                        }
                    )
                frame_idx += 1

        finally:
            cap.release()

        logger.info(f"[FrameExtractor] 共抽取 {len(frames)} 帧")
        return frames

    def cleanup(self) -> None:
        """删除临时帧目录中的所有文件。"""
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir), ignore_errors=True)
            logger.info(f"[FrameExtractor] 已清理临时目录: {self.temp_dir}")

    def get_video_info(self, video_path: str) -> dict:
        """获取视频基本信息。

        Args:
            video_path: 视频文件路径。

        Returns:
            包含以下字段的字典::

                {
                    "duration":      float,  # 时长（秒）
                    "fps":           float,  # 帧率
                    "width":         int,    # 宽度（像素）
                    "height":        int,    # 高度（像素）
                    "total_frames":  int,    # 总帧数
                }

        Raises:
            FileNotFoundError: 视频文件不存在。
            RuntimeError:      OpenCV 无法打开视频。
        """
        video_path = str(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps
        finally:
            cap.release()

        info = {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }
        logger.debug(f"[FrameExtractor] 视频信息: {info}")
        return info
