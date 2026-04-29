"""
llama-server 阶段化进程管理器（phase-swap）。

48GB 显存无法同时常驻 VL-32B(~22GB) + VL-7B(~5.5GB) + Script-32B(~22GB) + 解码/编码缓冲。
本模块按 pipeline 阶段拉起/关闭对应的 llama-server 子进程，避免显存超额。

阶段划分（默认）：
- Phase ``vl``:     启动 vl_server (32B) + vl_server_fast (7B)
- Phase ``script``: 关闭 vl 进程，启动 script_server (32B)

调用方在不同 Stage 之间显式调用 ``ensure_phase("vl")`` / ``ensure_phase("script")``。
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from utils.llm_client import LlamaCppClient


_LLAMA_SERVER_BIN_CANDIDATES = [
    "llama-server",
    "llama.cpp/build/bin/llama-server",
    "./llama.cpp/build/bin/llama-server",
]


def _resolve_llama_server() -> str:
    for candidate in _LLAMA_SERVER_BIN_CANDIDATES:
        # 绝对/相对路径直接判断
        if "/" in candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
        # PATH 查找
        from shutil import which
        if which(candidate):
            return candidate
    raise FileNotFoundError(
        "未找到 llama-server，可执行 scripts/build_llamacpp.sh 编译，"
        "或在 PATH 中放置 llama-server。"
    )


class ServerManager:
    """按 phase 启停 llama-server 进程。"""

    def __init__(self, config: dict, log_dir: str = "/tmp") -> None:
        self._config = config
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._current_phase: Optional[str] = None
        try:
            self._llama_bin = _resolve_llama_server()
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            self._llama_bin = "llama-server"  # 延迟到实际启动时再失败

    # ------------------------------------------------------------------
    # 阶段控制
    # ------------------------------------------------------------------

    async def ensure_phase(self, phase: str) -> None:
        """切换到指定阶段。重复调用同一 phase 是幂等的。"""
        if self._current_phase == phase:
            return
        logger.info(f"[ServerManager] 切换阶段: {self._current_phase} → {phase}")

        # 关掉与目标阶段无关的进程
        wanted = set(self._phase_processes(phase))
        to_stop = [name for name in self._procs if name not in wanted]
        for name in to_stop:
            self._stop_process(name)

        # 启动缺失的进程
        for name in wanted:
            if name not in self._procs:
                self._start_process(name)

        # 等待就绪
        await self._wait_ready(list(wanted))
        self._current_phase = phase
        logger.success(f"[ServerManager] 阶段 {phase} 就绪")

    def stop_all(self) -> None:
        for name in list(self._procs.keys()):
            self._stop_process(name)
        self._current_phase = None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_processes(phase: str) -> list[str]:
        if phase == "vl":
            return ["vl_server", "vl_server_fast"]
        if phase == "script":
            return ["script_server"]
        if phase == "none":
            return []
        raise ValueError(f"未知 phase: {phase}")

    def _build_cmd(self, name: str) -> tuple[list[str], int]:
        cfg = self._config.get(name)
        if not cfg:
            raise KeyError(f"配置缺少 {name} 节")
        cmd = [
            self._llama_bin,
            "--model", cfg["model_path"],
            "--host", str(cfg.get("host", "127.0.0.1")),
            "--port", str(cfg["port"]),
            "--n-gpu-layers", str(cfg.get("n_gpu_layers", 999)),
            "--ctx-size", str(cfg.get("ctx_size", 4096)),
            "--batch-size", str(cfg.get("batch_size", 512)),
            "--parallel", str(cfg.get("parallel", 2)),
            "--log-disable",
        ]
        if cfg.get("mmproj_path"):
            cmd += ["--mmproj", cfg["mmproj_path"]]
        if cfg.get("mlock"):
            cmd += ["--mlock"]
        return cmd, int(cfg["port"])

    def _start_process(self, name: str) -> None:
        cmd, port = self._build_cmd(name)
        log_file = self._log_dir / f"ai_video_{name}.log"
        logger.info(
            f"[ServerManager] 启动 {name} (port {port}): "
            f"{' '.join(shlex.quote(c) for c in cmd)}"
        )
        f = log_file.open("ab")
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        self._procs[name] = proc

    def _stop_process(self, name: str) -> None:
        proc = self._procs.pop(name, None)
        if not proc:
            return
        if proc.poll() is not None:
            return
        logger.info(f"[ServerManager] 关闭 {name} (pid {proc.pid})")
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _wait_ready(self, names: list[str], timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        pending = list(names)
        while pending and time.time() < deadline:
            still_pending = []
            for name in pending:
                cfg = self._config.get(name) or {}
                port = cfg.get("port")
                if not port:
                    continue
                client = LlamaCppClient(f"http://localhost:{port}")
                try:
                    ok = await client.health_check()
                finally:
                    await client.close()
                if ok:
                    logger.success(f"[ServerManager] {name} 健康检查通过 :{port}")
                else:
                    still_pending.append(name)
            pending = still_pending
            if pending:
                await asyncio.sleep(2.0)
        if pending:
            raise RuntimeError(
                f"[ServerManager] 服务未在 {timeout}s 内就绪: {pending}"
            )
