"""构建检索索引，并提供命令行检索验证。

两份语料各建各的索引（理由见 retrieval/corpus.py）。默认两份都建，
缺哪份语料就跳过哪份——只跑了官网抓取、还没构建知识库时也能正常工作。

用法：
    python -m retrieval.build_index                    # 构建全部可用语料的索引
    python -m retrieval.build_index -c site            # 只建站点语料
    python -m retrieval.build_index -q "什么是GEO"      # 构建后立即试检索
    python -m retrieval.search "品牌方提供哪些服务"      # 用已有索引检索
"""

from __future__ import annotations

import argparse

from retrieval.corpus import CORPORA, Corpus, get
from retrieval.hybrid import HybridRetriever


def show(retriever: HybridRetriever, query: str, top_k: int = 5) -> None:
    print(f"\n查询：{query}")
    print("─" * 78)
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        print("  无结果")
        return
    for i, h in enumerate(hits, 1):
        c = h.chunk
        print(f"{i}. {c.chunk_id:<8} RRF={h.score:.4f}  [{h.sources}]")
        print(f"   {c.heading}")


def build_one(corpus: Corpus, queries: list[str], top_k: int) -> HybridRetriever | None:
    if not corpus.chunks_path.exists():
        print(f"⏭  跳过{corpus.label}：未找到 {corpus.chunks_path.name}")
        return None

    print(f"\n▶ 构建{corpus.label}索引…")
    r = HybridRetriever.build(corpus)
    print(f"  完成：{len(r.chunks)} 个 chunk → {corpus.index_path.name}")
    for q in queries:
        show(r, q, top_k)
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="构建索引 / 试检索")
    ap.add_argument("-q", "--query", nargs="*", help="构建后试跑的查询")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("-c", "--corpus", choices=list(CORPORA), help="只构建指定语料，默认全部")
    args = ap.parse_args()

    print("首次运行需下载 bge-small-zh-v1.5（约 95MB）…")
    targets = [get(args.corpus)] if args.corpus else list(CORPORA.values())
    built = [build_one(c, args.query or [], args.top_k) for c in targets]

    if not any(built):
        print("\n没有任何语料可建索引。先运行 python -m kb.chunk 或 python -m website.chunk。")


if __name__ == "__main__":
    main()
