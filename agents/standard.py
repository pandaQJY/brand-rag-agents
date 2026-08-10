"""A 档判据：把品宣稿中的自述标准，变成审计官网的尺子。

这是知识库链路与 Agent 链路真正的咬合点，也是全系统说服力最强的一环。

背景：初版诊断把「官网无 JSON-LD」写成「AI 无从识别实体」——这站不住，
因为大模型读纯文本毫无障碍，JSON-LD 对 AI 检索管线的加权作用并无公开确证。
换判据之后结论才立得住：**品牌方自己的方法论把 JSON-LD 列为 GEO 网站基建的
首要能力（品宣稿 P29），而官网 52 页覆盖率为 0**。

这不是「我建议你加 schema」，而是「按你们自己写的标准，官网没做到」——
除非承认方法论有问题，否则无法反驳。

标准全部从品宣稿原文抽取，不由本模块编造：每条 Standard 都带 chunk_id 出处，
供报告标注引用，与写作链路的 [Pxx] 溯源机制共用同一套锚点。

尚未构建知识库时自动降级为空标准集，诊断退回仅 B/C 档，不影响运行。

用法：
    python -m agents.standard
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.config import CHUNKS_JSONL
from kb.chunk import load_chunks
from kb.models import Chunk

# scan.py 的 finding key → (在品宣稿中检索的关键词, 该标准要求什么)
# 只映射品宣稿确有主张的项。找不到原文支撑的，宁可留在 B 档。
STANDARD_RULES: dict[str, tuple[tuple[str, ...], str]] = {
    "jsonld": (
        ("JSON-LD", "结构化数据格式", "schema标签"),
        "官网应部署 JSON-LD 结构化数据，便于 AI 抓取、理解与引用",
    ),
    "internal_link": (
        ("优化内链结构", "页面权重", "内部信息关联增强"),
        "应优化内链结构与页面权重，增强内部信息关联",
    ),
    "faq_structure": (
        ("问答式搜索", "问答式"),
        "官网应支持问答式结构，便于 RAG 精准回答",
    ),
    "knowledge_graph": (
        ("内部知识图谱", "实体关联", "语义关系网络"),
        "应建立内部知识图谱，关联产品、FAQ、案例等实体",
    ),
}

_WS = re.compile(r"\s+")


@dataclass
class Standard:
    """一条来自品宣稿的自述标准。"""

    key: str
    requirement: str  # 该标准要求什么（本模块归纳）
    quote: str  # 品宣稿原文片段（不改写）
    source: str  # chunk_id，如 P29
    source_title: str = ""

    @property
    def citation(self) -> str:
        return f"[{self.source}]"


def _load_kb_chunks() -> list[Chunk]:
    """读取知识库的切分产物。未构建时返回空列表并降级。"""
    return load_chunks() if CHUNKS_JSONL.exists() else []


def _chunk_text(c: Chunk) -> str:
    return "\n".join(p for p in (c.title, c.text_layer, c.vision_text) if p)


def extract_standards() -> list[Standard]:
    """从品宣稿抽取可作为审计尺子的自述标准。"""
    chunks = _load_kb_chunks()
    if not chunks:
        return []

    found: list[Standard] = []
    for key, (keywords, requirement) in STANDARD_RULES.items():
        for c in chunks:
            text = _chunk_text(c)
            hit = next((k for k in keywords if k in text), None)
            if not hit:
                continue
            # 摘出关键词所在的一段原文作为引证，不做改写
            i = text.find(hit)
            quote = _WS.sub(" ", text[max(0, i - 60) : i + 90]).strip()
            found.append(
                Standard(
                    key=key,
                    requirement=requirement,
                    quote=quote,
                    source=c.chunk_id,
                    source_title=c.title,
                )
            )
            break  # 每条标准取首个命中出处即可
    return found


def apply_to(findings: list) -> list:
    """把 B 档发现升格为 A 档——当品宣稿对该项确有主张时。

    仅升格「有问题」的发现（severity 非 ok）：官网已达标的项无需拿标准去审。
    """
    std = {s.key: s for s in extract_standards()}
    if not std:
        return findings

    for f in findings:
        s = std.get(f.key)
        if not s or f.severity == "ok":
            continue
        f.grade = "A"
        f.detail = (
            f"与品牌方自述标准冲突。品宣稿 {s.citation}《{s.source_title}》主张："
            f"{s.requirement}。原文：「{s.quote}」。"
            f"—— 按品牌方自己的方法论，此项应当做到而官网未做到。"
        )
    return findings


def main() -> None:
    stds = extract_standards()
    if not stds:
        print(f"未找到知识库语料（{CHUNKS_JSONL}），诊断将降级为仅 B/C 档。")
        return

    print(f"从品宣稿抽出 {len(stds)} 条自述标准：\n")
    for s in stds:
        print(f"■ {s.key}  ←  {s.citation}《{s.source_title[:30]}》")
        print(f"   要求：{s.requirement}")
        print(f"   原文：{s.quote[:110]}\n")

    from agents.scan import scan

    findings = apply_to(scan())
    upgraded = [f for f in findings if f.grade == "A"]
    print(f"{'─' * 60}\n升格为 A 档的发现：{len(upgraded)} 条\n")
    for f in upgraded:
        print(f"🔴 [A档] {f.title}")
        print(f"    {f.fact}")
        print(f"    {f.detail}\n")


if __name__ == "__main__":
    main()
