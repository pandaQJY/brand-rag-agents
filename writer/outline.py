"""大纲规划：把写作主题拆成小节，并为每个小节生成检索查询。

这是「大纲先行」流水线的第一步，也是本方案与朴素 RAG 的主要分野。
朴素做法是用主题做一次检索，让所有小节共享一份证据集——结果是每一节
都不精准。这里改为：先规划大纲，每个小节自带 2-3 条查询，各自检索。

关键设计：**把资料库目录一并交给规划器**。
模型看得见资料的实际标题措辞，才能把「怎么衡量效果」这类宽问题改写成
「AI可见性监控 追踪指标」——这正是纯语义检索跨不过去的鸿沟。

用法：
    python -m writer.outline "为什么中国出海品牌需要进行GEO优化"
    python -m writer.outline "品牌方能提供什么服务" --type 产品介绍
"""

from __future__ import annotations

import argparse

from kb.chunk import load_chunks
from kb.models import Chunk
from core.llm import chat, load_prompt
from pydantic import BaseModel, Field, field_validator
from writer.content_types import CONTENT_TYPES, get_spec

__all__ = ["CONTENT_TYPES", "Outline", "Section", "build_catalog", "parse_outline", "plan"]


def _as_list(v):
    """把模型偶尔返回的单个字符串归一化为列表。

    JSON mode 只保证输出是合法 JSON，不保证类型符合 schema——实测模型
    会把 queries 写成 "查询A 查询B" 而非数组。这类偏差用校验重试纠正
    并不可靠（两次重试仍失败），在解析边界容错更稳。
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.replace("；", ";").split(";") if s.strip()]
    return v


class Section(BaseModel):
    heading: str = Field(description="小节标题")
    intent: str = Field(default="", description="这一节要回答的问题")
    queries: list[str] = Field(default_factory=list, description="该节的检索查询")

    _norm_queries = field_validator("queries", mode="before")(_as_list)


class Outline(BaseModel):
    title: str
    lead: str = ""
    sections: list[Section]
    faq: list[str] = Field(default_factory=list)

    _norm_faq = field_validator("faq", mode="before")(_as_list)


CATALOG_HINT_CHARS = 48


def _hint(chunk: Chunk) -> str:
    """取正文开头几行作为标题的补充线索。

    标题不总能说明这一页有什么。典型例子：P12 标题是《GEO时代到来：AI投毒乱象》，
    看不出它是全篇唯一定义了「GEO（生成式引擎优化）」的页——而它的正文首行
    正是「什么是GEO？」。只给标题会让规划器错过这类页，补上开头两行即可，
    且完全确定性、不花 API 成本。
    """
    lines = [ln.strip() for ln in chunk.text_layer.splitlines() if ln.strip()][:2]
    hint = " / ".join(lines)
    return hint[:CATALOG_HINT_CHARS] + ("…" if len(hint) > CATALOG_HINT_CHARS else "")


def build_catalog(chunks: list[Chunk] | None = None) -> str:
    """把资料库压成一份目录，供规划器了解素材边界与措辞。

    给编号、章节、标题与正文线索——足够模型判断"有没有这方面资料"，
    又不至于把整个语料塞进 Prompt。
    """
    chunks = chunks if chunks is not None else [c for c in load_chunks() if c.retrievable]
    lines = []
    for c in chunks:
        sec = c.section or "前言"
        lines.append(f"{c.chunk_id}\t[{sec}]\t{c.title}\t※ {_hint(c)}")
    return "\n".join(lines)


def parse_outline(text: str) -> Outline:
    """解析行式大纲。

    不用 JSON 的原因：实测 qwen-max 在「数组 → 对象 → 数组」这样的三层嵌套上
    系统性失败——每个小节都会漏掉 queries 的方括号并吐出垃圾片段。而查询原文
    本身又常含全角冒号与引号（照抄自资料标题），进一步加剧转义负担。
    行式格式没有嵌套、无需转义，解析确定且零失败。
    """
    outline = Outline(title="", lead="", sections=[])
    current: Section | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue

        if line.startswith("## "):
            current = Section(heading=line[3:].strip())
            outline.sections.append(current)
            continue

        tag, sep, value = line.partition(":")
        if not sep:
            continue
        tag, value = tag.strip().upper(), value.strip()

        if tag == "TITLE":
            outline.title = value
        elif tag == "LEAD":
            outline.lead = value
        elif tag == "FAQ":
            if value:
                outline.faq.append(value)
        elif tag == "INTENT" and current is not None:
            current.intent = value
        elif tag == "Q" and current is not None and value:
            current.queries.append(value)

    if not outline.title:
        raise ValueError("大纲缺少 TITLE")
    if not outline.sections:
        raise ValueError("大纲未解析出任何小节")
    return outline


def plan(topic: str, content_type: str = "官网Blog", provider: str | None = None) -> Outline:
    """生成大纲，并校验小节数量落在该内容类型的规格区间内。

    为什么要校验：数量约束原先只写在 Prompt 里。实测同一条 FAQ 指令两次规划，
    一次 7 个问题、一次 3 个——而规格明写「6–10 个问题」。模型违反了自己收到的
    约束，而系统照单全收，直接拿 3 节的大纲往下跑完整条链路。

    处置方式是**重试一次而非报错**：小节偏少不影响正确性，只影响完整度，
    为此中断整条链路不值当。重试仍不达标就带警告继续——把判断留给人，
    与第 3.5 层「标出来交给人」的处置一致。
    """
    spec = get_spec(content_type)
    prompt = load_prompt(
        "writer/outline",
        topic=topic,
        content_type=content_type,
        section_word=spec.section_word,
        type_guide=spec.outline_guide,
        catalog=build_catalog(),
    )

    outline = None
    for attempt in range(2):
        raw = chat(
            [{"role": "user", "content": prompt}], provider=provider, task="outline", max_tokens=2048
        )
        outline = parse_outline(raw)
        n = len(outline.sections)
        if spec.min_sections <= n <= spec.max_sections:
            return outline
        if attempt == 0:
            print(
                f"⚠ 大纲规划出 {n} 个{spec.section_word}，规格要求 "
                f"{spec.min_sections}–{spec.max_sections} 个，重试一次…"
            )

    print(
        f"⚠ 重试后仍为 {len(outline.sections)} 个{spec.section_word}"
        f"（规格 {spec.min_sections}–{spec.max_sections}），按现有大纲继续。"
    )
    return outline


def main() -> None:
    ap = argparse.ArgumentParser(description="写作大纲规划")
    ap.add_argument("topic", help="写作主题")
    ap.add_argument("--type", default="官网Blog", choices=CONTENT_TYPES, help="内容类型")
    ap.add_argument("--provider", help="覆盖默认供应商（qwen / deepseek）")
    args = ap.parse_args()

    outline = plan(args.topic, args.type, args.provider)

    print(f"\n标题：{outline.title}")
    print(f"导语：{outline.lead}\n")
    for i, s in enumerate(outline.sections, 1):
        print(f"{i}. {s.heading}")
        print(f"   意图：{s.intent}")
        for q in s.queries:
            print(f"   查询 → {q}")
        print()
    if outline.faq:
        print("FAQ：")
        for q in outline.faq:
            print(f"  · {q}")


if __name__ == "__main__":
    main()
