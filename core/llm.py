"""模型供应商适配层。

写作链路与 Agent 链路共用这一层——两者对模型的要求不同（写作要事实纪律，
Agent 要判断力），但差异应当落在 `Provider.task_models` 的选型表里，
而不是两份各自演化的调用代码。

DeepSeek 与通义千问（兼容模式）都提供 OpenAI 兼容接口，因此统一用 openai SDK，
仅切换 base_url 与模型名。换供应商只改 .env，不动业务代码。

对外暴露三个函数：
    chat()        纯文本对话
    vision()      图像理解（仅 qwen 提供，DeepSeek 官方 API 无视觉能力）
    chat_json()   结构化输出 + Pydantic 校验 + 失败重试
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from core.config import BASE_DIR, BRAND_ALIAS, BRAND_NAME, PROMPT_DIR

load_dotenv(BASE_DIR / ".env")

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    env_key: str
    chat_model: str
    vision_model: str | None = None
    # 按任务覆盖模型。五个环节的要求差异很大，不该共用一个模型：
    #   generate 要中文文笔，verify 要判定精度，intent 只是抽两个字段。
    task_models: dict[str, str] = field(default_factory=dict)

    def model_for(self, task: str | None = None) -> str:
        return self.task_models.get(task or "", self.chat_model)


PROVIDERS: dict[str, Provider] = {
    "qwen": Provider(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        chat_model="qwen-max",
        vision_model="qwen-vl-max",  # 视觉仅此一路
        task_models={
            # 实测（固定大纲与证据，仅换生成模型）：
            #   qwen-max    套话多，且违反 Prompt 明令禁止的「本文将探讨」句式
            #   qwen3-max   开篇即上带引用的硬事实，事实纪律良好
            #   deepseek-v3 文笔最佳，但把 61.66 亿写成 80 亿、凭空造出 300%
            # 本系统第一优先级是不得编造数据，故选 qwen3-max。
            "generate": "qwen3-max",
            # 意图解析只抽两个字段，用最快最便宜的即可
            "intent": "qwen-turbo",
        },
    ),
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        chat_model="deepseek-chat",
        vision_model=None,  # DeepSeek 官方 API 无视觉能力
    ),
}

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")


class LLMError(RuntimeError):
    pass


def get_provider(name: str | None = None) -> Provider:
    key = (name or DEFAULT_PROVIDER).lower()
    if key not in PROVIDERS:
        raise LLMError(f"未知供应商 {key!r}，可选：{list(PROVIDERS)}")
    return PROVIDERS[key]


def get_client(provider: str | None = None) -> tuple[OpenAI, Provider]:
    p = get_provider(provider)
    api_key = os.getenv(p.env_key)
    if not api_key:
        raise LLMError(f"环境变量 {p.env_key} 未设置。请复制 .env.example 为 .env 并填入。")
    return OpenAI(api_key=api_key, base_url=p.base_url, timeout=180.0), p


def load_prompt(name: str, **fields: str) -> str:
    """读取 prompts/ 下的提示词文件并填充占位符。

    Prompt 独立成文件而非硬编码：Prompt 的迭代频率远高于代码，
    分离后调整措辞无需改动、也无需重新审读业务逻辑。

    name 带子目录前缀，如 "writer/generate"、"agents/geo_auditor"。

    品牌名由配置注入而非写死在 Prompt 里——换一个分析对象时只改 .env，
    不必逐个文件替换。调用方显式传入的同名字段优先。
    """
    path = PROMPT_DIR / (name if name.endswith(".md") else f"{name}.md")
    text = path.read_text(encoding="utf-8")
    merged = {"brand": BRAND_NAME, "brand_alias": BRAND_ALIAS, **fields}
    for k, v in merged.items():
        text = text.replace("{" + k + "}", v)
    return text


def _retry(fn, attempts: int = 3, base_delay: float = 2.0):
    """对限流与瞬时故障做指数退避重试。"""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 供应商异常类型不统一
            last = e
            msg = str(e).lower()
            fatal = any(w in msg for w in ("invalid_api_key", "unauthorized", "not found", "invalid model"))
            if fatal or i == attempts - 1:
                break
            time.sleep(base_delay * (2**i))
    raise LLMError(f"调用失败（重试 {attempts} 次）：{last}") from last


def chat(
    messages: list[dict],
    *,
    provider: str | None = None,
    model: str | None = None,
    task: str | None = None,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """task 用于按环节选模型；显式传 model 优先级更高。"""
    client, p = get_client(provider)
    resp = _retry(
        lambda: client.chat.completions.create(
            model=model or p.model_for(task), messages=messages, max_tokens=max_tokens, **kwargs
        )
    )
    return resp.choices[0].message.content or ""


def _encode_image(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def vision(
    image_path: str | Path,
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 2048,
) -> tuple[str, dict]:
    """图像理解。返回 (文本, token 用量)。"""
    client, p = get_client(provider)
    if not p.vision_model:
        raise LLMError(
            f"供应商 {p.name!r} 无视觉能力。视觉转写请使用 qwen（在 .env 中设置 LLM_PROVIDER=qwen）。"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    resp = _retry(
        lambda: client.chat.completions.create(
            model=model or p.vision_model, messages=messages, max_tokens=max_tokens
        )
    )
    usage = {
        "in": resp.usage.prompt_tokens if resp.usage else 0,
        "out": resp.usage.completion_tokens if resp.usage else 0,
    }
    return (resp.choices[0].message.content or ""), usage


def _extract_json(text: str) -> str:
    """从可能带 ``` 围栏的回复中抠出 JSON 主体。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    start = min((i for i in (t.find("{"), t.find("[")) if i != -1), default=0)
    return t[start:].strip()


def chat_json(
    messages: list[dict],
    schema: type[T],
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 4096,
    attempts: int = 4,
) -> T:
    """结构化输出，带两种针对性的失败恢复策略。

    JSON mode 只保证「输出是合法 JSON」，不保证符合 schema。实测中出现两类失败，
    成因不同，恢复方式也应不同：

    1. **JSON 语法崩坏**（json.JSONDecodeError）——模型在嵌套结构里丢了括号。
       此时把残缺输出回传让它「修正」反而会顺着错误继续写，
       **重新生成一次**比修复更有效。

    2. **结构不符 schema**（ValidationError）——JSON 合法但字段类型或层级不对。
       模型看得懂校验报错，**带着错误信息修复**成功率高。
    """
    base = list(messages)
    convo = base
    last_err = ""

    for attempt in range(attempts):
        raw = chat(
            convo,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return schema.model_validate_json(_extract_json(raw))

        except json.JSONDecodeError as e:
            # 语法崩坏：丢弃上下文，重新生成
            last_err = f"JSON 语法错误：{e}"
            convo = base

        except ValidationError as e:
            # 结构不符：把报错回传，让模型定点修复
            last_err = str(e)[:600]
            if attempt < attempts - 1:
                convo = base + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"上面的输出不符合要求的结构，校验报错如下：\n{last_err}\n"
                            f"请仅输出修正后的完整 JSON，不要任何解释文字。"
                        ),
                    },
                ]

    raise LLMError(f"结构化输出失败（已尝试 {attempts} 次）：{last_err}")
