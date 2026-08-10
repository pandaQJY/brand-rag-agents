"""语料抽象：让同一套混合检索服务于两份来源不同的语料。

系统里有两份语料，它们的**引用锚点不同**：

    知识库语料   品宣稿切分而成，锚点是页码       [P29]
    站点语料     官网抓取切分而成，锚点是 URL     https://…/blog/xxx

锚点不同意味着**不能混进一个索引**——检索出一条证据却分不清该标 [P29]
还是标 URL，引用溯源就失去意义。因此两份语料各自建索引，由 Corpus 描述
「一份语料在磁盘上的三个文件 + 怎么把它读出来」。

检索层只依赖 Indexable 协议，不依赖具体的 Chunk 类型。这样 kb 与 website
两个包互不知晓，新增第三份语料（如客户案例库）也只需再声明一个 Corpus。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.config import (
    CHUNKS_JSONL,
    KB_IDMAP,
    KB_INDEX,
    SITE_CHUNKS_JSONL,
    SITE_IDMAP,
    SITE_INDEX,
)


@runtime_checkable
class Indexable(Protocol):
    """可被检索的最小单元。kb.models.Chunk 与 website.chunk.SiteChunk 都满足它。"""

    chunk_id: str
    retrievable: bool

    def index_text(self) -> str:
        """送入向量库与 BM25 的文本形态。"""
        ...

    @property
    def cite(self) -> str:
        """作为证据被引用时的标记。"""
        ...

    @property
    def heading(self) -> str:
        """一行摘要，用于检索结果展示。"""
        ...


@dataclass(frozen=True)
class Corpus:
    """一份语料：它在磁盘上的位置，以及怎么把它读出来。

    loader 用可调用对象而非直接 import，是为了避免 retrieval 包在导入时
    就拉起 kb 与 website 两条链路——BM25-only 的场景不该为此付出代价。
    """

    name: str
    label: str
    chunks_path: Path
    index_path: Path
    idmap_path: Path
    loader: Callable[[], list[Indexable]]

    def load_chunks(self) -> list[Indexable]:
        return self.loader()

    @property
    def is_built(self) -> bool:
        return self.chunks_path.exists() and self.index_path.exists()


def _load_kb() -> list[Indexable]:
    from kb.chunk import load_chunks

    return load_chunks()


def _load_site() -> list[Indexable]:
    from website.chunk import load_site_chunks

    return load_site_chunks()


KB = Corpus(
    name="kb",
    label="品宣稿知识库",
    chunks_path=CHUNKS_JSONL,
    index_path=KB_INDEX,
    idmap_path=KB_IDMAP,
    loader=_load_kb,
)

SITE = Corpus(
    name="site",
    label="官网站点语料",
    chunks_path=SITE_CHUNKS_JSONL,
    index_path=SITE_INDEX,
    idmap_path=SITE_IDMAP,
    loader=_load_site,
)

CORPORA: dict[str, Corpus] = {c.name: c for c in (KB, SITE)}


def get(name: str) -> Corpus:
    if name not in CORPORA:
        raise KeyError(f"未知语料 {name!r}，可选：{list(CORPORA)}")
    return CORPORA[name]
