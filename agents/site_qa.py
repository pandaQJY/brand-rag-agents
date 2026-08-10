"""原文问答 Agent —— 回答「官网上关于 X 是怎么说的」。

**为什么需要一个形态不同的 Agent。**

另外四个 Agent 都是「通读全站 → 输出固定小节」的分析器。这个形态
结构上答不了指向具体内容的问题：用户问「GEO 和 SEO 有什么区别」，
分析器的八个小节里没有一节是「官网对某个概念是怎么讲的」——
**答案就在已抓取的页面里，分析这条路却取不出它。**

取出它需要的是检索，不是又一个分析器。系统已有一套实测过的混合检索
（向量 + BM25，RRF 融合），把它指向站点语料即可：
检索定位到具体小节 → 模型据此作答 → 标注页面 URL 供人核对。

与分析型 Agent 的三点不同：

1. **无前置依赖。** 它不需要品牌基线，只需要检索得到的原文。
2. **产出随问题而变**（question_dependent），因此不参与追问缓存复用。
3. **答不出来要说答不出来。** 分析型 Agent 面对的是「把材料读一遍」，
   总有话可说；问答面对的是具体问题，官网没讲就必须承认，
   否则就成了拿行业常识冒充官网表述——那正是本系统全程在防的事。

用法：
    python -m agents.site_qa "GEO 和 SEO 有什么区别"
"""

from __future__ import annotations

import argparse

from agents.base import Agent, Blackboard, register
from retrieval.corpus import SITE
from retrieval.hybrid import HybridRetriever

# 取几个小节作为证据。站点 chunk 中位 387 字，5 条约 2k 字，
# 足够覆盖一个问题的答案面，又不至于把无关内容一并塞给模型。
TOP_K = 5

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """惰性加载：只有真正问到原文时才付出加载向量模型的开销。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever.load(SITE)
    return _retriever


def retrieve_excerpts(question: str, top_k: int = TOP_K) -> list:
    return get_retriever().search(question, top_k=top_k)


def format_excerpts(hits: list) -> str:
    """把检索结果排成带出处的证据块。

    URL 必须逐条附上——它是站点语料的引用锚点，作用等同知识库的 [Pxx]。
    模型答完之后，人要能顺着 URL 点开核对。
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(
            f"【证据 {i}】{c.section_title or c.page_title}\n"
            f"来源：{c.cite}\n"
            f"{c.text}"
        )
    return "\n\n".join(blocks) if blocks else "（检索无结果）"


@register
class SiteQA(Agent):
    name = "site_qa"
    label = "原文问答"
    description = (
        "回答关于官网具体内容的问题——某个概念怎么定义、两个术语有什么区别、"
        "某项服务具体怎么讲。检索原文后据实作答并标注页面 URL。"
        "问「官网上是怎么说的」归它；问「官网写得好不好」归 GEO 诊断"
    )
    depends_on = ()
    prompt_file = "agents/site_qa"
    max_tokens = 1200
    question_dependent = True  # 产出取决于问题本身，追问时不可复用

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        hits = retrieve_excerpts(bb.question)
        # 检索结果同时写进 Blackboard 的事实区，供界面展示「答案基于哪几段原文」
        bb.facts["site_qa_hits"] = hits
        return {"question": bb.question, "excerpts": format_excerpts(hits)}


def main() -> None:
    ap = argparse.ArgumentParser(description="原文问答：官网上关于 X 是怎么说的")
    ap.add_argument("question")
    ap.add_argument("-k", "--top-k", type=int, default=TOP_K)
    ap.add_argument("--evidence-only", action="store_true", help="只看检索到的原文，不调模型")
    args = ap.parse_args()

    hits = retrieve_excerpts(args.question, args.top_k)
    print(f"\n检索到 {len(hits)} 段原文：\n" + "─" * 72)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.chunk.heading}   RRF={h.score:.4f} [{h.sources}]")
        print(f"   {h.chunk.cite}")

    if args.evidence_only:
        return

    bb = Blackboard(question=args.question)
    result = SiteQA().run(bb)
    print("\n" + "═" * 72)
    print(result.text)


if __name__ == "__main__":
    main()
