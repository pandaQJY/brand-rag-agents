"""命令行检索工具，用于验证检索质量。

用法：
    python -m retrieval.search "品牌方提供哪些 GEO 服务"
    python -m retrieval.search "GEO 和 SEO 的区别" -k 3 --show-text
    python -m retrieval.search --compare "AI投毒"        # 对比两路与融合结果
    python -m retrieval.search "GEO 和 SEO 的区别" -c site  # 在官网语料里检索
"""

from __future__ import annotations

import argparse

from retrieval.corpus import CORPORA, get
from retrieval.hybrid import HybridRetriever


def cmd_search(r: HybridRetriever, query: str, top_k: int, show_text: bool) -> None:
    print(f"\n查询：{query}　（语料：{r.corpus.label}）")
    print("═" * 78)
    for i, h in enumerate(r.search(query, top_k=top_k), 1):
        c = h.chunk
        print(f"{i}. {c.chunk_id:<8} RRF={h.score:.4f}   [{h.sources}]")
        print(f"   {c.heading}")
        print(f"   出处：{c.cite}")
        if show_text:
            body = c.index_text().replace("\n", " ")[:220]
            print(f"   {body}…")
        print()


def cmd_compare(r: HybridRetriever, query: str, top_k: int) -> None:
    """并排展示向量路、BM25 路与 RRF 融合结果，用于观察融合行为。"""
    dense = r._vec.search(query, top_k)
    sparse = r._bm25.search(query, top_k)
    fused = r.search(query, top_k=top_k)

    print(f"\n查询：{query}")
    print("═" * 78)
    print(f"{'#':<3}{'向量检索（语义）':<26}{'BM25（关键词）':<26}{'RRF 融合':<20}")
    print("─" * 78)
    for i in range(top_k):
        d = f"{dense[i][0].chunk_id} {dense[i][1]:.3f}" if i < len(dense) else "—"
        s = f"{sparse[i][0].chunk_id} {sparse[i][1]:.2f}" if i < len(sparse) else "—"
        f = f"{fused[i].chunk.chunk_id} {fused[i].score:.4f}" if i < len(fused) else "—"
        both = "  ← 两路共同命中" if i < len(fused) and fused[i].dense_rank and fused[i].sparse_rank else ""
        print(f"{i + 1:<3}{d:<26}{s:<26}{f:<20}{both}")


def main() -> None:
    ap = argparse.ArgumentParser(description="检索验证工具")
    ap.add_argument("query", help="查询语句")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--show-text", action="store_true", help="显示 chunk 正文片段")
    ap.add_argument("--compare", action="store_true", help="对比两路检索与融合结果")
    ap.add_argument("-c", "--corpus", default="kb", choices=list(CORPORA), help="检索哪份语料")
    args = ap.parse_args()

    r = HybridRetriever.load(get(args.corpus))
    if args.compare:
        cmd_compare(r, args.query, args.top_k)
    else:
        cmd_search(r, args.query, args.top_k, args.show_text)


if __name__ == "__main__":
    main()
