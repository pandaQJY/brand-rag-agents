"""稀疏检索：jieba 分词 + BM25。

负责精确命中专有名词——品牌名、产品名、人名、技术术语。
这类词在向量空间里往往被稀释（品牌名与「公司」语义相近），
但用户查某个专名时要的就是字面命中，因此稀疏路不可省。
"""

from __future__ import annotations

import re

import jieba
from rank_bm25 import BM25Okapi

from retrieval.corpus import Indexable

# 中文停用词：高频但无区分度，参与打分只会稀释信号
_STOPWORDS = {
    "的", "了", "和", "是", "在", "有", "与", "为", "对", "等", "及", "或",
    "我们", "可以", "通过", "进行", "以及", "这个", "那个", "一个", "什么",
    "如何", "怎么", "哪些", "以", "被", "把", "让", "从", "到", "中", "上",
}
_TOKEN_OK = re.compile(r"[一-鿿]|[a-zA-Z0-9]")


def tokenize(text: str) -> list[str]:
    """分词并过滤。保留中文、英文与数字，丢掉标点与停用词。"""
    tokens = []
    for tok in jieba.lcut(text.lower()):
        tok = tok.strip()
        if not tok or tok in _STOPWORDS or not _TOKEN_OK.search(tok):
            continue
        tokens.append(tok)
    return tokens


class BM25Index:
    """基于 rank_bm25 的稀疏索引。两份语料都只有几十到几百条，
    全量内存构建即可，无需持久化——建索引的耗时远低于加载开销。"""

    def __init__(self, chunks: list[Indexable]):
        self.chunks = chunks
        self.corpus = [tokenize(c.index_text()) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Indexable, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k] if s > 0]
