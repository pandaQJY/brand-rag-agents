"""混合检索：向量 + BM25，用 RRF 融合。

为什么用 RRF 而不是加权求和：
余弦相似度落在 [0,1]，BM25 分数无上界且随语料波动，两者量纲不同，
加权求和必须先做归一化，而归一化系数会随语料变化失效。
RRF 只看**排名**不看分数，因此无需调权重、无需归一化——
工程上稳定得多，这也是它成为混合检索默认方案的原因。

    RRF(d) = Σ 1 / (k + rank_i(d))

k 是平滑常数（惯例取 60），作用是压低头部名次的绝对优势，
使得「在两路都排中游」的文档能胜过「只在一路排第一」的文档——
这正是混合检索想要的行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.bm25 import BM25Index
from retrieval.corpus import KB, Corpus, Indexable
from retrieval.embedder import VectorIndex

RRF_K = 60


@dataclass
class Hit:
    """一条检索结果，保留两路的原始名次以便调试与展示。"""

    chunk: Indexable
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def sources(self) -> str:
        tags = []
        if self.dense_rank is not None:
            tags.append(f"向量#{self.dense_rank}")
        if self.sparse_rank is not None:
            tags.append(f"BM25#{self.sparse_rank}")
        return " + ".join(tags)


@dataclass
class HybridRetriever:
    """一份语料上的混合检索器。

    语料由 Corpus 指定，默认是品宣稿知识库；站点语料传 corpus=SITE
    即可复用同一套检索逻辑——这正是原文问答 Agent 的实现基础。
    """

    chunks: list[Indexable] = field(default_factory=list)
    corpus: Corpus = KB
    _bm25: BM25Index | None = None
    _vec: VectorIndex | None = None

    @classmethod
    def load(cls, corpus: Corpus = KB) -> HybridRetriever:
        """加载语料与索引。向量索引缺失时降级为纯 BM25。

        降级而非报错，是因为两条路的成本不对称：BM25 在内存里现建即可，
        而向量索引需要先跑一次 build_index。只抓了官网还没建索引时，
        原文问答仍应能回答问题——检索质量下降，但功能不塌。
        降级会打印提示，不静默。
        """
        chunks = [c for c in corpus.load_chunks() if c.retrievable]
        vec = None
        if corpus.index_path.exists() and corpus.idmap_path.exists():
            vec = VectorIndex.load(chunks, corpus.index_path, corpus.idmap_path)
        else:
            print(
                f"⚠ 未找到{corpus.label}的向量索引（{corpus.index_path.name}），"
                f"本次仅用 BM25。构建：python -m retrieval.build_index -c {corpus.name}"
            )
        return cls(chunks=chunks, corpus=corpus, _bm25=BM25Index(chunks), _vec=vec)

    @property
    def dense_available(self) -> bool:
        return self._vec is not None

    @classmethod
    def build(cls, corpus: Corpus = KB) -> HybridRetriever:
        """构建并落盘向量索引。"""
        chunks = [c for c in corpus.load_chunks() if c.retrievable]
        vec = VectorIndex.build(chunks)
        vec.save(corpus.index_path, corpus.idmap_path)
        return cls(chunks=chunks, corpus=corpus, _bm25=BM25Index(chunks), _vec=vec)

    def search(self, query: str, top_k: int = 5, pool: int = 12) -> list[Hit]:
        """两路各取 pool 条，RRF 融合后返回 top_k。

        向量路不可用时只走 BM25——RRF 对单路退化为按名次排序，
        结果顺序与纯 BM25 一致，无需为降级情形另写一条分支。
        """
        dense = self._vec.search(query, pool) if self._vec else []
        sparse = self._bm25.search(query, pool)

        dense_rank = {c.chunk_id: i + 1 for i, (c, _) in enumerate(dense)}
        sparse_rank = {c.chunk_id: i + 1 for i, (c, _) in enumerate(sparse)}

        scores: dict[str, float] = {}
        for ranks in (dense_rank, sparse_rank):
            for cid, r in ranks.items():
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + r)

        by_id = {c.chunk_id: c for c in self.chunks}
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Hit(
                chunk=by_id[cid],
                score=s,
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
            )
            for cid, s in ordered
        ]
