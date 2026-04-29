"""
视频抽帧工具

提供两种抽帧路径：

1. :meth:`FrameExtractor.extract` —— 旧路径，OpenCV CPU 解码，按 fps 抽帧。
   保留用于 ``scene_detection.enabled=false`` 的传统逐帧分析。
2. :meth:`FrameExtractor.extract_thumbnails_nvdec` —— **新路径**，用
   ``ffmpeg -hwaccel cuda`` GPU 解码 + ``scale=`` 缩放到 360p，按目标 fps 输出
   低分辨率缩略图序列。4K 2h 电影实测 ≤ 90s。

新版主流水线（电影解说）应使用 NVDEC 路径喂给 PySceneDetect。
"""

import os
import shutil
import subprocess
from pathlib import Path

import cv2
from loguru import logger


class FrameExtractor:
    """视频抽帧工具。"""

    def __init__(self, temp_dir: str = "/tmp/ai_video_frames") -> None:
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[FrameExtractor] 临时目录: {self.temp_dir}")

    # ------------------------------------------------------------------
    # 新路径：ffmpeg + NVDEC 抽缩略图（推荐）
    # ------------------------------------------------------------------

    def extract_thumbnails_nvdec(
        self,
        video_path: str,
        fps: float = 2.0,
        size: tuple[int, int] = (640, 360),
        out_subdir: str = "thumbs",
        quality: int = 4,
    ) -> list[dict]:
        """用 ffmpeg + NVDEC 解码并输出低分缩略图序列。

        ffmpeg 命令::

            ffmpeg -hwaccel cuda -i in.mp4 \\
                -vf "fps=2,scale=640:360" -q:v 4 thumbs/%08d.jpg

        Args:
            video_path: 输入视频路径。
            fps:        采样帧率（帧/秒）。2.0 对 PySceneDetect 已足够。
            size:       (width, height) 缩放分辨率。
            out_subdir: 子目录名（相对于 ``temp_dir``）。
            quality:    ffmpeg ``-q:v`` 参数（2-5，越小越好）。

        Returns:
            列表，每项::

                {
                    "frame_idx": int,        # 缩略图序号（1-based 与文件名一致）
                    "timestamp": float,      # 对应时间戳（秒）
                    "path": str,             # JPEG 路径
                }
        """
        video_path = str(video_path)
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("[FrameExtractor] 系统未安装 ffmpeg")

        out_dir = self.temp_dir / out_subdir
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        w, h = int(size[0]), int(size[1])
        vf = f"fps={fps},scale={w}:{h}"
        cmd_cuda = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-hwaccel", "cuda",
            "-i", video_path,
            "-vf", vf,
            "-q:v", str(quality),
            str(out_dir / "thumb_%08d.jpg"),
        ]
        cmd_cpu = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", vf,
            "-q:v", str(quality),
            str(out_dir / "thumb_%08d.jpg"),
        ]
        logger.info(
            f"[FrameExtractor] NVDEC 抽缩略图: fps={fps} size={w}x{h} → {out_dir}"
        )
        proc = subprocess.run(cmd_cuda, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(
                f"[FrameExtractor] NVDEC 解码失败，回退到 CPU。stderr: "
                f"{proc.stderr.strip()[:200]}"
            )
            subprocess.run(cmd_cpu, check=True)

        files = sorted(out_dir.glob("thumb_*.jpg"))
        if not files:
            raise RuntimeError("[FrameExtractor] 未生成任何缩略图，检查输入视频")

        period = 1.0 / float(fps)
        thumbs: list[dict] = []
        for i, fp in enumerate(files, start=1):
            ts = (i - 0.5) * period
            thumbs.append({
                "frame_idx": i,
                "timestamp": round(ts, 3),
                "path": str(fp),
            })
        logger.success(f"[FrameExtractor] 共生成 {len(thumbs)} 张缩略图")
        return thumbs

    def extract_frame_at(
        self,
        video_path: str,
        timestamp: float,
        out_path: str,
        size: tuple[int, int] | None = None,
        use_cuda: bool = True,
    ) -> str:
        """在指定时间戳抽一张高质量代表帧（用于 VL 精分）。

        Args:
            video_path: 输入视频路径。
            timestamp: 时间戳（秒）。
            out_path:  输出 JPEG 路径。
            size:      可选 (width, height) 缩放，None 则保持原分辨率。
            use_cuda:  是否使用 NVDEC。

        Returns:
            输出 JPEG 路径。
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if use_cuda:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{timestamp:.3f}", "-i", str(video_path), "-frames:v", "1"]
        if size:
            cmd += ["-vf", f"scale={int(size[0])}:{int(size[1])}"]
        cmd += ["-q:v", "3", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            # 回退到 CPU
            cmd_cpu = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{timestamp:.3f}", "-i", str(video_path),
                "-frames:v", "1",
            ]
            if size:
                cmd_cpu += ["-vf", f"scale={int(size[0])}:{int(size[1])}"]
            cmd_cpu += ["-q:v", "3", str(out_path)]
            subprocess.run(cmd_cpu, check=True)
        return str(out_path)

    # ------------------------------------------------------------------
    # 旧路径：OpenCV 抽帧（仅传统模式用）
    # ------------------------------------------------------------------

    def extract(
        self,
        video_path: str,
        fps_sample: float = 1.0,
        quality: int = 85,
    ) -> list[dict]:
        """OpenCV 抽帧（CPU 路径，供 legacy 模式）。"""
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
                    frames.append({
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "path": str(out_path),
                    })
                frame_idx += 1
        finally:
            cap.release()

        logger.info(f"[FrameExtractor] 共抽取 {len(frames)} 帧")
        return frames

    # ------------------------------------------------------------------
    # 元信息 / 清理
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir), ignore_errors=True)
            logger.info(f"[FrameExtractor] 已清理临时目录: {self.temp_dir}")

    def get_video_info(self, video_path: str) -> dict:
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
        return {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }
