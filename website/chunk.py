"""把抓取到的官网页面切成可检索、可引用的单元。

**切分粒度：按 h2 分节，而不是按页。**

知识库那边以页为单位，因为幻灯片本身就是设计好的语义块。网页不是——
一篇 5,760 字的 Blog 是一个连续文档，整页做成一个 chunk 会让检索命中它，
却无法指出「答案在哪一段」。而站点语料的用途恰恰是回答
「官网上关于 X 是怎么说的」，落点必须比一整页更细。

实测依据：首页 3,508 字、62 个标题，其中就包含「GEO 与 SEO 的区别」那组
FAQ。整页一个 chunk 时，这段被淹没在另外六十多个标题里；按 h2 切开后，
它成为独立单元，检索可以直接命中。

抓取产物的正文已把标题以 Markdown 形式内联（`## 二级标题`），
因此无需重新解析 HTML，按行扫描即可还原分节结构。

用法：
    python -m website.chunk
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field

from core.config import SITE_CHUNKS_JSONL
from website.fetch import load as load_pages
from website.models import SitePage

# 一节短于此字符数时并入上一节。孤立的小标题（如「更多」「联系我们」）
# 自成一个 chunk 只会稀释索引，与知识库把分节页并入后一页是同一考量。
MIN_SECTION_CHARS = 120

# 一节超过此长度时按段落二次切分，避免长 Blog 的单节盖过整页。
MAX_SECTION_CHARS = 1800


class SiteChunk(BaseModel):
    """站点语料的检索单元。引用锚点是 URL，不是页码。"""

    chunk_id: str = Field(description="如 S07 或 S07-2（同页第 2 节）")
    url: str
    page_title: str = ""
    section_title: str = Field(default="", description="本节的 h2 标题")
    text: str = ""
    retrievable: bool = True

    @property
    def path(self) -> str:
        return urlparse(self.url).path or "/"

    @property
    def char_count(self) -> int:
        return len(self.text)

    def index_text(self) -> str:
        """送入向量库与 BM25 的文本形态。

        URL 路径一并写入：`/blog/geo-provider-selection-pitfalls` 这类语义化
        路径本身携带主题词，对 BM25 那一路尤其有用。
        """
        parts = [f"【{self.chunk_id}】{self.page_title}".rstrip()]
        if self.section_title:
            parts.append(f"小节：{self.section_title}")
        parts.append(f"页面路径：{self.path}")
        parts.append(self.text)
        return "\n".join(p for p in parts if p)

    # ── retrieval.corpus.Indexable 协议 ──

    @property
    def cite(self) -> str:
        """站点证据的引用锚点是 URL——读者可以点开核对。"""
        return self.url

    @property
    def heading(self) -> str:
        title = self.section_title or self.page_title or self.path
        return f"《{title}》 · {self.path}"


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 二级标题把正文切成 [(小节标题, 正文)]。

    h2 之前的内容归入一个标题为空的前导节——首页的主张句、Blog 的导语
    都落在这里，它们往往是全页信息密度最高的一段，不能丢。
    """
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.split("\n"):
        if line.startswith("## ") and not line.startswith("### "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(t, "\n".join(body).strip()) for t, body in sections]


def _merge_short(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """把过短的节并入相邻节，避免产生无信息量的碎片 chunk。

    默认向后并（并入前一节），但**首节没有前一节**——而首节恰恰是最常见的
    碎片来源：许多列表页的前导节只有一个 h1 页面标题（`# 博客`、`# GEO中心`）。
    这类碎片若留在索引里，会以纯标题的身份与真正有内容的节竞争排名。
    因此首节过短时改为向前并入下一节。
    """
    merged: list[tuple[str, str]] = []
    for title, body in sections:
        if not body and not title:
            continue
        whole = f"{title}\n{body}".strip() if title else body
        if merged and len(whole) < MIN_SECTION_CHARS:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body}\n{whole}".strip())
        else:
            merged.append((title, body))

    # 首节过短且当时无前节可并 —— 改并入其后一节，标题沿用后一节的
    if len(merged) > 1 and len(merged[0][0]) + len(merged[0][1]) < MIN_SECTION_CHARS:
        head_title, head_body = merged.pop(0)
        head = f"{head_title}\n{head_body}".strip()
        next_title, next_body = merged[0]
        merged[0] = (next_title, f"{head}\n{next_body}".strip())

    return merged


def _split_long(title: str, body: str) -> list[tuple[str, str]]:
    """超长节按段落二次切分，标题沿用，保证每片仍知道自己属于哪一节。

    末片过短时并回前一片：按固定长度切分必然在结尾留下余数，
    这个余数不是一个语义单元，独立成 chunk 只会造出碎片。
    """
    if len(body) <= MAX_SECTION_CHARS:
        return [(title, body)]

    parts: list[str] = []
    buf: list[str] = []
    size = 0
    for para in body.split("\n"):
        if size + len(para) > MAX_SECTION_CHARS and buf:
            parts.append("\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        parts.append("\n".join(buf))

    if len(parts) > 1 and len(parts[-1]) < MIN_SECTION_CHARS:
        tail = parts.pop()
        parts[-1] = f"{parts[-1]}\n{tail}"

    return [(title, p) for p in parts]


def build_site_chunks(pages: list[SitePage]) -> list[SiteChunk]:
    chunks: list[SiteChunk] = []
    for i, page in enumerate(pages, 1):
        pieces: list[tuple[str, str]] = []
        for title, body in _merge_short(split_sections(page.text)):
            pieces.extend(_split_long(title, body))

        for j, (title, body) in enumerate(pieces, 1):
            if not body.strip():
                continue
            suffix = "" if len(pieces) == 1 else f"-{j}"
            chunks.append(
                SiteChunk(
                    chunk_id=f"S{i:02d}{suffix}",
                    url=page.url,
                    page_title=page.title,
                    section_title=title,
                    text=body,
                )
            )
    return chunks


def save_site_chunks(chunks: list[SiteChunk], path=SITE_CHUNKS_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_site_chunks(path=SITE_CHUNKS_JSONL) -> list[SiteChunk]:
    with open(path, encoding="utf-8") as f:
        return [SiteChunk.model_validate_json(ln) for ln in f if ln.strip()]


def main() -> None:
    pages = load_pages()
    chunks = build_site_chunks(pages)
    save_site_chunks(chunks)

    sizes = sorted(c.char_count for c in chunks)
    multi = len({c.url for c in chunks if "-" in c.chunk_id})
    print(f"切分完成：{len(pages)} 页 → {len(chunks)} 个 chunk  → {SITE_CHUNKS_JSONL}")
    print(f"  其中 {multi} 页被切成多节，其余整页成节")
    print(f"  体量：最小 {sizes[0]} / 中位 {sizes[len(sizes) // 2]} / 最大 {sizes[-1]} 字符")


if __name__ == "__main__":
    main()
