"""
显存监控与管理工具

提供 GPU 显存查询、阈值告警和强制垃圾回收功能，
帮助在 48GB 单卡环境下安全调度多个大模型。
"""

import gc
from typing import Optional

import torch
from loguru import logger


class VRAMManager:
    """GPU 显存状态监控与管理。

    依赖 PyTorch CUDA 接口，仅在 CUDA 可用时功能完整；
    CPU-only 环境下所有查询方法返回 0.0，不影响流程运行。
    """

    _GB = 1024 ** 3  # bytes per gigabyte

    def __init__(self, device_index: int = 0) -> None:
        """初始化显存管理器。

        Args:
            device_index: GPU 设备索引，默认 0（第一块 GPU）。
        """
        self._device = device_index
        self._cuda_available = torch.cuda.is_available()
        if not self._cuda_available:
            logger.warning("[VRAMManager] CUDA 不可用，显存管理功能降级为空操作。")

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_free_vram(self) -> float:
        """返回当前空闲显存（GB）。"""
        if not self._cuda_available:
            return 0.0
        free_bytes, _ = torch.cuda.mem_get_info(self._device)
        return free_bytes / self._GB

    def get_used_vram(self) -> float:
        """返回当前已用显存（GB）。"""
        if not self._cuda_available:
            return 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
        return (total_bytes - free_bytes) / self._GB

    def get_total_vram(self) -> float:
        """返回 GPU 总显存（GB）。"""
        if not self._cuda_available:
            return 0.0
        _, total_bytes = torch.cuda.mem_get_info(self._device)
        return total_bytes / self._GB

    # ------------------------------------------------------------------
    # 阈值检查
    # ------------------------------------------------------------------

    def check_threshold(self, threshold_gb: float = 2.0) -> bool:
        """检查空闲显存是否低于阈值。

        Args:
            threshold_gb: 阈值（GB），默认 2.0 GB。

        Returns:
            空闲显存低于阈值时返回 ``True``（表示紧张），否则 ``False``。
        """
        free = self.get_free_vram()
        is_tight = free < threshold_gb
        if is_tight:
            logger.warning(
                f"[VRAMManager] 显存告警：空闲 {free:.2f} GB < 阈值 {threshold_gb} GB"
            )
        return is_tight

    # ------------------------------------------------------------------
    # 主动释放
    # ------------------------------------------------------------------

    def force_gc(self) -> float:
        """强制执行 Python 垃圾回收并清空 CUDA 缓存。

        Returns:
            本次释放的显存量（GB）；CUDA 不可用时返回 0.0。
        """
        before = self.get_free_vram()
        gc.collect()
        if self._cuda_available:
            torch.cuda.empty_cache()
        after = self.get_free_vram()
        released = after - before
        logger.info(
            f"[VRAMManager] force_gc: 释放前 {before:.2f} GB 空闲 → "
            f"释放后 {after:.2f} GB 空闲（+{released:.2f} GB）"
        )
        return released

    # ------------------------------------------------------------------
    # 状态日志
    # ------------------------------------------------------------------

    def log_status(self) -> None:
        """打印当前显存使用情况到日志。"""
        if not self._cuda_available:
            logger.info("[VRAMManager] CUDA 不可用，无显存信息。")
            return
        free = self.get_free_vram()
        used = self.get_used_vram()
        total = self.get_total_vram()
        pct = (used / total * 100) if total > 0 else 0.0
        logger.info(
            f"[VRAMManager] GPU:{self._device}  "
            f"已用 {used:.1f} GB / 总计 {total:.1f} GB  "
            f"空闲 {free:.1f} GB  ({pct:.1f}% 已占用)"
        )
