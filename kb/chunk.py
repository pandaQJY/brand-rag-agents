"""切分：把逐页解析结果组织成检索单元 Chunk。

以页为基本语义单元——幻灯片本身就是设计好的语义块，页码天然是引用锚点。
在此之上做两件事：

1. 识别分节页（带大号装饰序号、自身几无内容），并入其后一页，
   避免产生"只有一个标题"的噪声 chunk；
2. 把分节页的标题作为章节名向后传播，让每个 chunk 都带章节上下文。

用法：
    python -m ingest.chunk
"""

from __future__ import annotations

import re

from core.config import CHUNKS_JSONL
from kb.models import Chunk, PageDoc
from kb.parse import load_pages

# 分节页上的大号装饰序号：内容形如 "01"，字号显著大于正文标题
_DECORATIVE = re.compile(r"^[\d\s.、,，/·\-—()（）]{1,4}$")
DIVIDER_MIN_FONT = 50.0

# 字数上限只是防御性兜底，不是主判据。
# 主判据是「超大号纯数字块唯一」——目录页有 01/02/03 三个，因此被排除。
# 此处防的是另一种情况：正文页用超大字号展示单个统计数字（如 "80%"）。
# 实测三个分节页为 15 / 27 / 61 字，取 120 留足余量。
DIVIDER_MAX_CHARS = 120


def is_divider(page: PageDoc) -> bool:
    """判定分节页。

    判据有二：存在**唯一**一个超大号纯数字块，且页面文字量很小。
    「唯一」这一条把目录页排除在外——目录同样有 01/02/03 大字，
    但它有三个，且承载实际信息。
    """
    big = [b for b in page.blocks if b.max_font >= DIVIDER_MIN_FONT and _DECORATIVE.match(b.text.strip())]
    return len(big) == 1 and page.char_count <= DIVIDER_MAX_CHARS


def is_toc(page: PageDoc) -> bool:
    """判定目录/议程页。

    目录页与分节页共用大号数字装饰，区别在数量：分节页一个，目录页多个。
    这类页罗列全部章节名——关键词覆盖率最高、实质内容为零，是检索的
    "关键词磁铁"：查任何主题都会命中，却提供不了任何可引用的事实。
    因此保留在数据里（便于人工查阅），但不进检索索引。
    """
    big = [b for b in page.blocks if b.max_font >= DIVIDER_MIN_FONT and _DECORATIVE.match(b.text.strip())]
    return len(big) >= 2


def assign_sections(pages: list[PageDoc]) -> dict[int, str]:
    """由分节页向后传播章节名，返回 页码 → 章节名。"""
    sections: dict[int, str] = {}
    current = ""
    for p in pages:
        if is_divider(p):
            seq = next(
                (b.text.strip() for b in p.blocks
                 if b.max_font >= DIVIDER_MIN_FONT and _DECORATIVE.match(b.text.strip())),
                "",
            )
            current = f"{seq} {p.title}".strip()
        sections[p.page_no] = current
    return sections


def _clean_text_layer(text: str) -> str:
    """去掉孤立的装饰序号行（分节页的 "01"/"02"），它们对检索是噪声。"""
    return "\n".join(ln for ln in text.splitlines() if not _DECORATIVE.match(ln.strip()))


# 视觉模型偶尔不遵守「无内容的小节整节省略」，改写成「（图中未出现…）」占位。
# 这类句子会被检索误命中（查"机构名"时匹配到"未出现机构名"），须剔除。
_EMPTY_NOTE = re.compile(r"^[（(].*(无|未出现|不适用|没有).*[）)]$")


def _drop_empty_sections(text: str) -> str:
    """删除只含占位说明的小节，连同其标题一并去掉。"""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("#"):
            body: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("#"):
                if lines[j].strip():
                    body.append(lines[j].strip())
                j += 1
            # 整节为空，或全部是占位说明 → 丢弃标题与内容
            if not body or all(_EMPTY_NOTE.match(b) for b in body):
                i = j
                continue
            out.append(line)
            out.extend(lines[i + 1 : j])
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip()


def _merge(pages: list[PageDoc], section: str) -> Chunk:
    """把一页或多页合成一个 Chunk。"""
    nos = [p.page_no for p in pages]
    cid = f"P{nos[0]:02d}" if len(nos) == 1 else f"P{nos[0]:02d}-{nos[-1]:02d}"
    # 分节页的标题即章节名，正文标题取最后一页的（信息量更大）
    title = next((p.title for p in reversed(pages) if p.title), "")
    text_layer = "\n".join(_clean_text_layer(p.text_layer) for p in pages if p.text_layer)
    vision_text = "\n".join(_drop_empty_sections(p.vision_text) for p in pages if p.vision_text)
    return Chunk(
        chunk_id=cid,
        page_nos=nos,
        section=section,
        title=title,
        retrievable=not any(is_toc(p) for p in pages),
        text_layer="\n".join(ln for ln in text_layer.splitlines() if ln.strip()),
        vision_text="\n".join(ln for ln in vision_text.splitlines() if ln.strip()),
        image_paths=[p.image_path for p in pages],
    )


def build_chunks(pages: list[PageDoc]) -> list[Chunk]:
    sections = assign_sections(pages)
    chunks: list[Chunk] = []
    pending: list[PageDoc] = []  # 待并入下一页的分节页

    for p in pages:
        if is_divider(p):
            pending.append(p)
            continue
        group = pending + [p]
        pending = []
        chunks.append(_merge(group, sections[p.page_no]))

    if pending:  # 分节页出现在末尾，无后续页可并
        chunks.append(_merge(pending, sections[pending[-1].page_no]))

    return chunks


def save_chunks(chunks: list[Chunk], path=CHUNKS_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks(path=CHUNKS_JSONL) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.model_validate_json(ln) for ln in f if ln.strip()]


def main() -> None:
    pages = load_pages()
    chunks = build_chunks(pages)
    save_chunks(chunks)

    dividers = [p.cite_id for p in pages if is_divider(p)]
    excluded = [c.chunk_id for c in chunks if not c.retrievable]
    sizes = sorted(c.char_count for c in chunks)
    by_section: dict[str, int] = {}
    for c in chunks:
        by_section[c.section or "（前言）"] = by_section.get(c.section or "（前言）", 0) + 1

    print(f"切分完成：{len(pages)} 页 → {len(chunks)} 个 chunk  → {CHUNKS_JSONL}")
    print(f"  分节页 {len(dividers)} 个 {dividers}，已并入后一页")
    print(f"  排除出索引 {len(excluded)} 个 {excluded}（目录页，无实质内容）")
    print(f"  体量：最小 {sizes[0]} / 中位 {sizes[len(sizes) // 2]} / 最大 {sizes[-1]} 字符")
    print("\n章节分布：")
    for sec, n in by_section.items():
        print(f"  {n:>2} 个 chunk   {sec}")
    print("\n合并的 chunk：")
    for c in chunks:
        if len(c.page_nos) > 1:
            print(f"  {c.chunk_id}  {c.char_count:>5}字  《{c.title}》")


if __name__ == "__main__":
    main()
