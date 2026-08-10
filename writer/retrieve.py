"""分节检索：按大纲逐节取证据。

朴素 RAG 用主题做一次检索，所有小节共享一份证据集——每一节都不精准。
这里每个小节用自己的 2-3 条查询各自检索，节内再用 RRF 把多条查询的结果
融成一份排序，最后取 top-k 作为该节的证据。

一个 chunk 可以同时服务多个小节（例如「四大核心技术能力」既支撑技术节
也支撑价值节），因此节间不做去重；但对外提供一份全局去重的证据集，
供事实核查环节比对使用。

用法：
    python -m writer.retrieve "为什么中国出海品牌需要进行GEO优化"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from kb.models import Chunk
from retrieval.hybrid import RRF_K, HybridRetriever, Hit
from writer.outline import Outline, Section, plan

EVIDENCE_PER_SECTION = 4


@dataclass
class SectionEvidence:
    section: Section
    hits: list[Hit] = field(default_factory=list)

    def context(self) -> str:
        return "\n\n".join(h.chunk.index_text() for h in self.hits)


@dataclass
class EvidencePack:
    outline: Outline
    sections: list[SectionEvidence] = field(default_factory=list)

    @property
    def chunks(self) -> list[Chunk]:
        """全局去重的证据集，保持首次出现顺序。"""
        seen: set[str] = set()
        out: list[Chunk] = []
        for se in self.sections:
            for h in se.hits:
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    out.append(h.chunk)
        return out

    def full_context(self) -> str:
        """全部证据拼成一份，供事实核查使用。"""
        return "\n\n".join(c.index_text() for c in self.chunks)

    @property
    def coverage(self) -> str:
        n = len(self.chunks)
        return f"{n} 份证据 / 覆盖 {len(self.sections)} 个小节"


def retrieve_section(
    retriever: HybridRetriever,
    section: Section,
    top_k: int = EVIDENCE_PER_SECTION,
    pool: int = 8,
) -> SectionEvidence:
    """一个小节的多条查询各自检索，再用 RRF 融合成一份排序。

    与两路融合同理：不同查询给出的分数不可直接比较，只有名次可比。
    """
    scores: dict[str, float] = {}
    best_hit: dict[str, Hit] = {}

    for q in section.queries:
        for rank, hit in enumerate(retriever.search(q, top_k=pool), 1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            # 保留名次最靠前的那次命中，其 sources 信息更有代表性
            if cid not in best_hit or rank < 2:
                best_hit.setdefault(cid, hit)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    hits = [
        Hit(
            chunk=best_hit[cid].chunk,
            score=s,
            dense_rank=best_hit[cid].dense_rank,
            sparse_rank=best_hit[cid].sparse_rank,
        )
        for cid, s in ordered
    ]
    return SectionEvidence(section=section, hits=hits)


def build_evidence(
    outline: Outline,
    retriever: HybridRetriever | None = None,
    top_k: int = EVIDENCE_PER_SECTION,
) -> EvidencePack:
    retriever = retriever or HybridRetriever.load()
    return EvidencePack(
        outline=outline,
        sections=[retrieve_section(retriever, s, top_k) for s in outline.sections],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="按大纲分节检索证据")
    ap.add_argument("topic", help="写作主题")
    ap.add_argument("--type", default="官网Blog")
    ap.add_argument("-k", "--top-k", type=int, default=EVIDENCE_PER_SECTION)
    ap.add_argument("--show-text", action="store_true", help="显示证据正文片段")
    args = ap.parse_args()

    print("规划大纲…")
    outline = plan(args.topic, args.type)
    print("分节检索…\n")
    pack = build_evidence(outline, top_k=args.top_k)

    print(f"《{outline.title}》   {pack.coverage}")
    print("═" * 76)
    for i, se in enumerate(pack.sections, 1):
        print(f"\n{i}. {se.section.heading}")
        for q in se.section.queries:
            print(f"   查询 → {q}")
        print("   证据：")
        for h in se.hits:
            print(f"     {h.chunk.chunk_id:<8} {h.chunk.title[:34]:<36} [{h.sources}]")
            if args.show_text:
                print(f"       {h.chunk.index_text()[:130].replace(chr(10), ' ')}…")

    print("\n" + "═" * 76)
    print(f"全局去重证据集：{[c.chunk_id for c in pack.chunks]}")


if __name__ == "__main__":
    main()
