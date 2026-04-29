"""
llama.cpp OpenAI 兼容接口异步客户端
支持文本对话和多模态视觉理解

用法示例::

    client = LlamaCppClient("http://localhost:8080")
    response = await client.chat([{"role": "user", "content": "你好"}])
    await client.close()
"""

import asyncio
import base64
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger


class LlamaCppClient:
    """与 llama.cpp llama-server 通信的异步 HTTP 客户端。

    兼容 OpenAI `/v1/chat/completions` 接口，支持文本对话与图文多模态输入。
    """

    def __init__(self, base_url: str, timeout: int = 180) -> None:
        """初始化客户端。

        Args:
            base_url: llama-server 根地址，例如 ``http://localhost:8080``。
            timeout:  请求超时秒数，默认 180 秒（大模型推理可能较慢）。
        """
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """调用 ``/v1/chat/completions`` 接口，自动重试 3 次（指数退避）。

        Args:
            messages:    OpenAI 格式消息列表。
            temperature: 采样温度，0 为贪心解码。
            max_tokens:  最大生成 token 数。
            **kwargs:    其他传递给接口的参数（如 top_p、stop 等）。

        Returns:
            模型返回的文本内容。

        Raises:
            RuntimeError: 三次重试后仍失败时抛出。
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post("/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    f"[LlamaCppClient] 第 {attempt + 1} 次请求失败: {exc}，"
                    f"{wait}s 后重试…"
                )
                await asyncio.sleep(wait)
        raise RuntimeError(
            f"[LlamaCppClient] 三次重试均失败，最后错误: {last_error}"
        )

    async def vision_chat(
        self,
        prompt: str,
        image_path: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """多模态图文对话：将图片编码为 base64 后发送给 VL 模型。

        Args:
            prompt:      文本提示词。
            image_path:  本地图片路径（JPEG/PNG）。
            temperature: 采样温度，视觉任务通常用较低值（0.1）。
            max_tokens:  最大生成 token 数。

        Returns:
            模型对图片内容的文字描述。
        """
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in {"jpg", "jpeg"} else f"image/{suffix}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return await self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    async def vision_chat_b64(
        self,
        prompt: str,
        image_b64: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """与 vision_chat 相同，但直接接受 base64 字符串（避免重复读文件）。"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def health_check(self) -> bool:
        """检查服务是否就绪。

        Returns:
            服务健康返回 ``True``，否则返回 ``False``。
        """
        try:
            resp = await self._client.get("/health", timeout=10.0)
            return resp.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False

    async def close(self) -> None:
        """关闭底层 httpx AsyncClient，释放连接资源。"""
        await self._client.aclose()


def create_clients(config: dict) -> tuple["LlamaCppClient", "LlamaCppClient"]:
    """根据配置字典创建视觉理解和脚本生成两个客户端实例。

    Args:
        config: 从 ``config/model_config.yaml`` 解析的字典。

    Returns:
        ``(vl_client, script_client)`` 元组。
    """
    vl_port = config["vl_server"]["port"]
    script_port = config["script_server"]["port"]
    vl_client = LlamaCppClient(f"http://localhost:{vl_port}")
    script_client = LlamaCppClient(f"http://localhost:{script_port}")
    logger.info(
        f"[create_clients] VL 客户端 → :{vl_port}  "
        f"Script 客户端 → :{script_port}"
    )
    return vl_client, script_client


def create_fast_client(config: dict) -> "LlamaCppClient | None":
    """创建 VL-7B 快速粗筛客户端。

    若配置中没有 vl_server_fast 节，则返回 None（调用方降级为 32B）。
    """
    fast_cfg = config.get("vl_server_fast")
    if fast_cfg is None:
        logger.warning("配置中无 vl_server_fast，粗筛将使用 vl_server（32B）")
        return None
    return LlamaCppClient(f"http://localhost:{fast_cfg['port']}")
