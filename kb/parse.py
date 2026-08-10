"""文字通道：PDF 文字层提取、阅读顺序恢复、页面渲染。

这是离线链路的第一步，不依赖任何模型 API。产物：
    data/pages.jsonl        每页一条 PageDoc（vision_text 为空，待 vision.py 填充）
    data/page_images/*.jpg  每页渲染图，供视觉通道使用

阅读顺序问题：PDF 文字块的存储顺序不等于阅读顺序。品宣稿常见"三栏卡片"
排版，若按存储顺序直接拼接，会把三栏的第一行串成一句。这里先把块按视觉
行分组，行内从左到右、行间从上到下重排。

运行：python -m ingest.parse
"""

from __future__ import annotations

import re

import fitz  # PyMuPDF

from core.config import (
    JPEG_QUALITY,
    PAGE_IMAGE_DIR,
    PAGES_JSONL,
    RENDER_FORMAT,
    RENDER_MAX_EDGE,
    ROW_OVERLAP_RATIO,
    SOURCE_PDF,
    TITLE_TOP_RATIO,
    ensure_dirs,
)
from kb.models import PageDoc, TextBlock

_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    """折叠空白但保留换行结构。"""
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _extract_blocks(page: fitz.Page) -> list[dict]:
    """抽出页面上的文本块，携带 bbox 与最大字号。顺序为 PDF 存储顺序。"""
    raw = page.get_text("dict")["blocks"]
    out: list[dict] = []

    for blk in raw:
        if blk.get("type") != 0:  # 0 = 文本块，1 = 图像块
            continue

        text_parts: list[str] = []
        max_font = 0.0
        for line in blk.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            if line_text.strip():
                text_parts.append(line_text)
            for s in spans:
                max_font = max(max_font, float(s.get("size", 0)))

        text = _clean("\n".join(text_parts))
        if not text:
            continue

        out.append({"text": text, "bbox": tuple(map(float, blk["bbox"])), "max_font": max_font})

    return out


def _vertical_overlap_ratio(a: tuple, b: tuple) -> float:
    """两个 bbox 垂直方向的重叠占较矮者高度的比例。"""
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    overlap = bottom - top
    if overlap <= 0:
        return 0.0
    shorter = min(a[3] - a[1], b[3] - b[1])
    return overlap / shorter if shorter > 0 else 0.0


def _order_blocks(raw_blocks: list[dict]) -> list[TextBlock]:
    """恢复阅读顺序：先按视觉行分组，行内左→右，行间上→下。

    贪心分行：把块按 y0 排序后依次考察，若与当前行任一成员的垂直重叠
    超过阈值，则并入该行，否则另起一行。这样能正确处理并列卡片排版。
    """
    if not raw_blocks:
        return []

    by_top = sorted(raw_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    rows: list[list[dict]] = []
    for blk in by_top:
        placed = False
        for row in rows:
            if any(_vertical_overlap_ratio(blk["bbox"], m["bbox"]) >= ROW_OVERLAP_RATIO for m in row):
                row.append(blk)
                placed = True
                break
        if not placed:
            rows.append([blk])

    # 行内左→右；行间按行内最小 y0 上→下
    rows.sort(key=lambda r: min(m["bbox"][1] for m in r))
    ordered: list[TextBlock] = []
    for row in rows:
        for blk in sorted(row, key=lambda b: b["bbox"][0]):
            ordered.append(
                TextBlock(
                    order=len(ordered),
                    text=blk["text"],
                    bbox=blk["bbox"],
                    max_font=round(blk["max_font"], 1),
                )
            )
    return ordered


_DECORATIVE = re.compile(r"^[\d\s.、,，/·\-—()（）]{1,4}$")


def _is_decorative(text: str) -> bool:
    """章节分隔页上的大号装饰数字（"01"/"02"）字号最大但不是标题。"""
    return bool(_DECORATIVE.match(text.strip()))


def _detect_title(blocks: list[TextBlock], page_height: float) -> str:
    """标题 = 页面上部区域内字号最大的非装饰块。

    该 PDF 字号分层清晰（标题 32pt / 小标题 18pt / 正文 12pt），
    字号判据比位置判据更可靠；限定上部区域是为了排除页脚大字装饰。
    分节页的大号序号（60-96pt）字号反而更大，须按内容形态剔除。
    """
    if not blocks:
        return ""

    cutoff = page_height * TITLE_TOP_RATIO
    upper = [b for b in blocks if b.bbox[1] <= cutoff] or blocks

    candidates = [b for b in upper if not _is_decorative(b.text)] or upper
    best = max(candidates, key=lambda b: (b.max_font, -b.bbox[1]))

    # 标题通常是单行短句；过长说明命中了正文，宁可留空
    first_line = best.text.splitlines()[0].strip()
    return first_line if len(first_line) <= 60 else ""


def _render_page(page: fitz.Page, out_path) -> None:
    """把页面渲染为图片，长边缩放到 RENDER_MAX_EDGE。"""
    long_edge = max(page.rect.width, page.rect.height)
    zoom = RENDER_MAX_EDGE / long_edge if long_edge else 1.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    if RENDER_FORMAT == "jpg":
        pix.save(out_path, jpg_quality=JPEG_QUALITY)
    else:
        pix.save(out_path)


def parse_pdf(pdf_path=SOURCE_PDF, render: bool = True) -> list[PageDoc]:
    """解析整份 PDF，返回逐页 PageDoc。"""
    ensure_dirs()
    doc = fitz.open(pdf_path)
    pages: list[PageDoc] = []

    for idx, page in enumerate(doc):
        page_no = idx + 1
        blocks = _order_blocks(_extract_blocks(page))
        text_layer = "\n".join(b.text for b in blocks)
        title = _detect_title(blocks, page.rect.height)

        image_name = f"p{page_no:02d}.{RENDER_FORMAT}"
        if render:
            _render_page(page, PAGE_IMAGE_DIR / image_name)

        # 标题已单独成字段，正文里去掉重复的首行
        body = text_layer
        if title and body.startswith(title):
            body = body[len(title) :].lstrip("\n")

        pages.append(
            PageDoc(
                page_no=page_no,
                width=page.rect.width,
                height=page.rect.height,
                title=title,
                text_layer=body,
                blocks=blocks,
                image_count=len(page.get_images()),
                char_count=len(_WS.sub("", text_layer)),
                image_path=f"page_images/{image_name}",
            )
        )

    doc.close()
    return pages


def save_pages(pages: list[PageDoc], path=PAGES_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(p.model_dump_json() + "\n")


def load_pages(path=PAGES_JSONL) -> list[PageDoc]:
    with open(path, encoding="utf-8") as f:
        return [PageDoc.model_validate_json(ln) for ln in f if ln.strip()]


def main() -> None:
    pages = parse_pdf()
    save_pages(pages)

    total_chars = sum(p.char_count for p in pages)
    total_imgs = sum(p.image_count for p in pages)
    sparse = [p for p in pages if p.is_text_sparse]
    untitled = [p for p in pages if not p.title]

    print(f"解析完成：{len(pages)} 页 → {PAGES_JSONL}")
    print(f"  文字层总字符   {total_chars:,}")
    print(f"  平均每页       {total_chars // len(pages)}")
    print(f"  内嵌图片总数   {total_imgs}")
    print(f"  文字层近空页   {len(sparse)}  {[p.cite_id for p in sparse]}")
    print(f"  未识别出标题   {len(untitled)}  {[p.cite_id for p in untitled]}")
    print(f"  渲染图         {PAGE_IMAGE_DIR}")

    print("\n前 8 页标题抽检：")
    for p in pages[:8]:
        flag = "  ← 依赖视觉通道" if p.is_text_sparse else ""
        print(f"  {p.cite_id}  {p.char_count:>4}字 {p.image_count:>3}图  {p.title or '（无标题）'}{flag}")


if __name__ == "__main__":
    main()
