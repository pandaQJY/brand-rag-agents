"""检索质量评测。

用一组人工标注的「查询 → 应命中页」评估检索效果，输出 Hit@k 与 MRR。
有了它，任何检索侧改动（换模型、调 RRF 参数、增删过滤规则）都能量化比较，
而不是靠翻几条结果拍脑袋。

标注依据：逐页人工阅读品宣稿，为每个查询标出确实含有答案的页。

用法：
    python -m retrieval.eval          # 评测混合检索
    python -m retrieval.eval --all    # 同时评测向量、BM25、混合三路
"""

from __future__ import annotations

import argparse

from retrieval.hybrid import HybridRetriever

# (查询, 应命中的 chunk_id 集合)
# chunk_id 与页码一致；合并 chunk 用 P08-09 形式
EVAL_SET: list[tuple[str, set[str]]] = [
    ("AI投毒是什么", {"P12"}),
    ("GEO 的四大核心技术能力", {"P27"}),
    ("监控系统能看到哪些数据指标", {"P32"}),
    ("有哪些客户成功案例", {"P37", "P38", "P39", "P40"}),
    ("商业模式和收费方式", {"P41"}),
    ("主要服务哪些行业", {"P34-35"}),
    ("对招商银行的 GEO 应用设想", {"P19"}),
    ("品牌价值蒸馏模型是什么", {"P25"}),
    ("问题预测能力", {"P28"}),
    ("GEO 原生网站基建能力", {"P29"}),
    ("媒体分发渠道怎么选", {"P31"}),
    ("AI 电商的增长趋势", {"P10"}),
    ("公司被哪些媒体报道过", {"P03", "P13"}),
    ("AI 搜索正在成为新的流量入口", {"P08-09"}),
    ("为什么选择 LLM 原生 GEO", {"P04"}),
    ("金融行业的 AI 信任入口", {"P16", "P17"}),
]

# 上面一组查询多与页标题接近，区分度不足（三路 Hit@5 均为 100%）。
# 下面这组刻意加难，贴近真实写作场景：
#   · 主题式提问，与任何标题都不字面重合
#   · 专名查询，考察 BM25 对向量的补足
HARD_SET: list[tuple[str, set[str]]] = [
    # 主题式：Blog 大纲会生成的那种宽问题
    ("出海品牌在 AI 时代面临什么挑战", {"P11", "P12", "P22", "P23"}),
    ("怎么衡量优化之后有没有效果", {"P32"}),
    ("为什么品牌会被 AI 忽略", {"P22", "P23", "P25"}),
    ("内容要怎么改才能被大模型采纳", {"P30", "P26"}),
    ("和传统 SEO 有什么不一样", {"P04", "P26"}),
    # 专名式：向量易稀释，BM25 应占优
    # P13 是创始团队页，P03 中人名仅出现在新闻标题里
    ("毛慧娜", {"P13", "P03"}),
    ("招商银行", {"P17", "P19"}),
    ("生成式引擎推荐原理", {"P26"}),
    # OPC 全篇仅出现一次且未展开缩写，无语义近邻——向量几乎必然失手
    ("OPC 模式是什么", {"P41"}),
]


def evaluate(search_fn, name: str, k: int = 5, dataset=None) -> dict:
    """计算 Hit@1 / Hit@k / MRR。"""
    dataset = dataset if dataset is not None else EVAL_SET
    hit1 = hitk = 0
    rr_total = 0.0
    misses: list[str] = []

    for query, expected in dataset:
        ids = [c.chunk_id for c in search_fn(query, k)]
        if ids and ids[0] in expected:
            hit1 += 1
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in expected), None)
        if rank:
            hitk += 1
            rr_total += 1.0 / rank
        else:
            misses.append(f"{query} → 期望 {sorted(expected)}，实得 {ids[:3]}")

    n = len(dataset)
    return {
        "name": name,
        "hit1": hit1 / n,
        "hitk": hitk / n,
        "mrr": rr_total / n,
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="检索质量评测")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--all", action="store_true", help="对比向量 / BM25 / 混合三路")
    ap.add_argument("--hard", action="store_true", help="使用高难度评测集")
    args = ap.parse_args()

    r = HybridRetriever.load()
    k = args.top_k

    runners = [(lambda q, n: [h.chunk for h in r.search(q, top_k=n)], "混合 (RRF)")]
    if args.all:
        runners = [
            (lambda q, n: [c for c, _ in r._vec.search(q, n)], "向量 (bge)"),
            (lambda q, n: [c for c, _ in r._bm25.search(q, n)], "BM25"),
            *runners,
        ]

    dataset = HARD_SET if args.hard else EVAL_SET
    label = "高难度集" if args.hard else "基础集"
    print(f"{label}：{len(dataset)} 条查询，top_k={k}\n")
    print(f"{'检索方式':<14}{'Hit@1':>8}{'Hit@' + str(k):>8}{'MRR':>8}")
    print("─" * 40)
    results = []
    for fn, name in runners:
        res = evaluate(fn, name, k, dataset)
        results.append(res)
        print(f"{res['name']:<14}{res['hit1']:>7.1%}{res['hitk']:>8.1%}{res['mrr']:>8.3f}")

    final = results[-1]
    if final["misses"]:
        print(f"\n{final['name']} 未命中的查询：")
        for m in final["misses"]:
            print(f"  · {m}")


if __name__ == "__main__":
    main()
