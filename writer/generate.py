"""生成：依据分节证据写出带引用标记的文章。

同时实现幻觉拦截的第 2、3 层（均为纯代码、零 API 成本）：
    第 2 层  引用格式解析——正则抽出全文 [Pxx]，定位到具体句子
    第 3 层  引用有效性校验——比对本次证据集，编号不存在即判为捏造引用

第 1 层（Prompt 硬约束）在 prompts/generate.md 中；
第 4 层（LLM 事实核查）在 writer/verify.py 中。

用法：
    python -m writer.generate "为什么中国出海品牌需要进行GEO优化"
    python -m writer.generate "品牌方能提供什么服务" --type 产品介绍 -o article.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.llm import chat, load_prompt
from writer.content_types import CONTENT_TYPES, get_spec
from writer.outline import Outline, plan
from writer.retrieve import EvidencePack, build_evidence

# chunk_id 形如 P12 或 P08-09
CITE_RE = re.compile(r"\[(P\d{2}(?:-\d{2})?)\]")
# 句子切分：中文句末标点，用于把引用定位到所属句子
SENT_SPLIT = re.compile(r"(?<=[。！？；])")


@dataclass
class Sentence:
    """一个句子及其携带的引用标记。"""

    text: str
    cites: list[str] = field(default_factory=list)

    @property
    def has_cite(self) -> bool:
        return bool(self.cites)


@dataclass
class Article:
    markdown: str
    outline: Outline
    evidence_ids: list[str]

    @property
    def cited_ids(self) -> list[str]:
        """正文中出现过的全部引用编号，去重保序。"""
        seen, out = set(), []
        for m in CITE_RE.findall(self.markdown):
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    @property
    def invalid_cites(self) -> list[str]:
        """第 3 层拦截：引用了证据集里不存在的编号 = 捏造引用。"""
        return [c for c in self.cited_ids if c not in self.evidence_ids]

    @property
    def unused_evidence(self) -> list[str]:
        """检索到但未被引用的证据，用于判断检索是否过量。"""
        return [e for e in self.evidence_ids if e not in self.cited_ids]

    def sentences(self) -> list[Sentence]:
        """第 2 层：把正文切成句子并附上各自的引用。"""
        out: list[Sentence] = []
        for line in self.markdown.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for piece in SENT_SPLIT.split(s):
                piece = piece.strip()
                if piece:
                    out.append(Sentence(text=piece, cites=CITE_RE.findall(piece)))
        return out

    @property
    def stats(self) -> dict:
        sents = self.sentences()
        body = CITE_RE.sub("", self.markdown)
        return {
            "字数": len(re.sub(r"\s|[#*`]", "", body)),
            "句子数": len(sents),
            "带引用句": sum(1 for s in sents if s.has_cite),
            "引用总数": len(CITE_RE.findall(self.markdown)),
            "引用页数": len(self.cited_ids),
            "证据数": len(self.evidence_ids),
            "未使用证据": len(self.unused_evidence),
        }


def _format_outline(outline: Outline) -> str:
    """把大纲渲染给生成器。

    intent 是给生成器看的写作指示，不是正文——实测模型会把「本节要回答：xxx」
    原样抄进文章开头。因此用括号包成明确的旁注口吻，并在 Prompt 中声明其性质。
    """
    lines = [f"标题：{outline.title}", f"导语应覆盖：{outline.lead}", ""]
    for i, s in enumerate(outline.sections, 1):
        lines.append(f"{i}. {s.heading}")
        if s.intent:
            lines.append(f"   （写作指示，不得写入正文）本节需交代：{s.intent}")
    if outline.faq:
        lines.append("")
        lines.append("FAQ 问题：")
        lines.extend(f"   - {q}" for q in outline.faq)
    return "\n".join(lines)


def _format_evidence(pack: EvidencePack) -> str:
    """按小节组织证据。同一份证据服务多节时会重复出现，这是有意的——
    让模型清楚每一节手上有什么，避免跨节挪用。"""
    blocks = []
    for i, se in enumerate(pack.sections, 1):
        blocks.append(f"### 供第 {i} 节《{se.section.heading}》使用")
        blocks.append(se.context() or "（本节未检索到证据）")
    return "\n\n".join(blocks)


def generate(
    pack: EvidencePack, content_type: str = "官网Blog", provider: str | None = None
) -> Article:
    spec = get_spec(content_type)
    prompt = load_prompt(
        "writer/generate",
        content_type=content_type,
        type_guide=spec.generate_guide,
        outline=_format_outline(pack.outline),
        evidence=_format_evidence(pack),
    )
    md = chat([{"role": "user", "content": prompt}], provider=provider, task="generate", max_tokens=4096)
    md = md.strip()
    if md.startswith("```"):
        md = md.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return Article(
        markdown=md,
        outline=pack.outline,
        evidence_ids=[c.chunk_id for c in pack.chunks],
    )


def write_article(topic: str, content_type: str = "官网Blog", provider: str | None = None) -> Article:
    outline = plan(topic, content_type, provider)
    pack = build_evidence(outline)
    return generate(pack, content_type, provider)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成带引用的文章")
    ap.add_argument("topic")
    ap.add_argument("--type", default="官网Blog", choices=CONTENT_TYPES)
    ap.add_argument("--provider")
    ap.add_argument("-o", "--out", help="保存为 Markdown 文件")
    args = ap.parse_args()

    print("规划大纲…")
    outline = plan(args.topic, args.type, args.provider)
    print("分节检索…")
    pack = build_evidence(outline)
    print(f"生成中（{len(pack.chunks)} 份证据）…\n")
    article = generate(pack, args.type, args.provider)

    print(article.markdown)
    print("\n" + "═" * 70)
    for k, v in article.stats.items():
        print(f"  {k}：{v}")

    if article.invalid_cites:
        print(f"\n  ⚠ 第 3 层拦截：捏造引用 {article.invalid_cites}")
    else:
        print("\n  ✓ 第 3 层通过：全部引用编号均存在于证据集")
    if article.unused_evidence:
        print(f"  · 未被引用的证据：{article.unused_evidence}")

    if args.out:
        Path(args.out).write_text(article.markdown, encoding="utf-8")
        print(f"\n  已保存 → {args.out}")


if __name__ == "__main__":
    main()
