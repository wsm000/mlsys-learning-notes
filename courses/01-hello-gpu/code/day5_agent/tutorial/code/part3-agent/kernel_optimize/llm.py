"""薄 LLM 客户端：一个 ``chat()``，直接用 OpenAI 兼容接口（LiteLLM 按前缀路由）。

教学要点：agent 的「大脑」就这么简单——把消息发给模型，拿回回复（可能带工具调用）。
模型配置走环境变量（或 ``~/.config/hello-gpu/kernel-agent.env``）：

    KERNEL_AGENT_MODEL=provider/model   # 如 openai/gpt-4o、deepseek/deepseek-chat
    KERNEL_AGENT_API_BASE=...           # 可选，openai 兼容端点
    KERNEL_AGENT_API_KEY=...            # 可选，统一密钥
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 在 import litellm 前关掉它拉远程 cost map 的请求（实验机网络不稳时会卡）。
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_MODEL_COST_MAP_URL", "")

import litellm  # noqa: E402

litellm.suppress_debug_info = True
litellm.set_verbose = False

DEFAULT_ENV_FILE = Path.home() / ".config" / "hello-gpu" / "kernel-agent.env"

# provider 前缀 → LiteLLM 认识的专用密钥环境变量。
_PROVIDER_KEY = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "azure": "AZURE_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """把 KEY=VALUE 的 env 文件塞进 os.environ（不覆盖已存在的）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _model_config() -> tuple[str, str | None]:
    _load_env_file()
    model = os.environ.get("KERNEL_AGENT_MODEL", "").strip()
    if not model or "/" not in model:
        raise RuntimeError(
            "缺少 KERNEL_AGENT_MODEL 环境变量（格式 provider/model，"
            "如 openai/gpt-4o 或 deepseek/deepseek-chat）。"
            "可写进 ~/.config/hello-gpu/kernel-agent.env。"
        )
    api_base = (
        os.environ.get("KERNEL_AGENT_API_BASE")
        or os.environ.get("KERNEL_AGENT_MODEL_URL")
        or None
    )
    api_key = os.environ.get("KERNEL_AGENT_API_KEY")
    if api_key:
        provider = model.split("/", 1)[0]
        if _PROVIDER_KEY.get(provider):
            os.environ.setdefault(_PROVIDER_KEY[provider], api_key)
        if api_base:  # openai 兼容端点也认 OPENAI_API_KEY
            os.environ.setdefault("OPENAI_API_KEY", api_key)
    return model, api_base


def chat(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
) -> Any:
    """调一次模型，返回 message 对象（.content / .tool_calls）。"""
    model, api_base = _model_config()
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if api_base:
        kwargs["api_base"] = api_base
    api_key = os.environ.get("KERNEL_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    # 默认关闭 Radeon Cloud / Qwen 风格的 thinking，避免占满 token 且 content 为空。
    # 显式 KERNEL_AGENT_EXTRA_BODY 会整体替换默认值：DeepSeek 官方接口使用
    # {"thinking": {"type": "disabled"}}，不能与 enable_thinking 混发。
    extra_body: dict[str, Any] = {"enable_thinking": False}
    raw_extra = os.environ.get("KERNEL_AGENT_EXTRA_BODY", "").strip()
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
            if isinstance(parsed, dict):
                extra_body = parsed
        except json.JSONDecodeError:
            pass
    kwargs["extra_body"] = extra_body
    response = litellm.completion(**kwargs)
    return response.choices[0].message
