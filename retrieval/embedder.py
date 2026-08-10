"""稠密检索：bge-small-zh-v1.5 + FAISS。

选择本地模型而非云端 Embedding API 的理由：
克隆仓库后无需配置任何 key 即可跑通检索与评测，把「要花钱才能验证」
这道门槛从检索链路上彻底移除。bge-small-zh 仅 ~95MB，
中文语义检索表现足够，首次运行自动下载。

索引文件路径由调用方传入而非写死在模块里——系统有两份语料
（品宣稿、官网），各自一套索引，见 retrieval/corpus.py。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retrieval.corpus import Indexable

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# bge 系列要求查询侧加指令前缀，文档侧不加。省略会明显掉召回。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_model = None


def get_model():
    """惰性加载，避免仅做 BM25 检索时也付出加载开销。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_docs(texts: list[str]) -> np.ndarray:
    vecs = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def encode_query(text: str) -> np.ndarray:
    vec = get_model().encode([QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype="float32")


class VectorIndex:
    """FAISS 内积索引。向量已归一化，内积即余弦相似度。"""

    def __init__(self, chunks: list[Indexable], index=None):
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(cls, chunks: list[Indexable]) -> VectorIndex:
        import faiss

        vecs = encode_docs([c.index_text() for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(chunks, index)

    def save(self, index_path: Path, idmap_path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(index_path))
        idmap_path.write_text("\n".join(c.chunk_id for c in self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, chunks: list[Indexable], index_path: Path, idmap_path: Path) -> VectorIndex:
        """按 idmap 记录的顺序重排 chunks——FAISS 只存向量，
        第 i 行对应哪个 chunk 全靠这份顺序表，错位即全盘错。"""
        import faiss

        by_id = {c.chunk_id: c for c in chunks}
        order = idmap_path.read_text(encoding="utf-8").split("\n")
        return cls([by_id[i] for i in order if i in by_id], faiss.read_index(str(index_path)))

    def search(self, query: str, top_k: int = 10) -> list[tuple[Indexable, float]]:
        scores, idx = self.index.search(encode_query(query), min(top_k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
