"""知识库的数据结构。

PageDoc 是离线阶段的中心对象：一页 PDF 对应一个 PageDoc，
文字通道与视觉通道的产物分字段存放，便于核查时定位事实来源。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """页面上的一个文本块，保留位置与字号以便恢复阅读顺序、识别标题。"""

    order: int = Field(description="恢复阅读顺序后的序号，从 0 开始")
    text: str
    bbox: tuple[float, float, float, float] = Field(description="(x0, y0, x1, y1)")
    max_font: float = Field(description="块内最大字号，用于标题识别")

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PageDoc(BaseModel):
    """一页的完整解析结果。

    text_layer 与 vision_text 分开存放，不做提前合并——
    事实核查时需要知道某条陈述来自可提取文字还是来自图像转写。
    """

    page_no: int = Field(description="页码，从 1 开始，即引用锚点 [Pxx] 中的数字")
    width: float
    height: float

    title: str = Field(default="", description="按字号推断的页面标题")
    text_layer: str = Field(default="", description="文字通道产物，已恢复阅读顺序")
    blocks: list[TextBlock] = Field(default_factory=list)

    image_count: int = Field(default=0, description="页内嵌图片数")
    char_count: int = Field(default=0, description="文字层字符数，去空白后")
    image_path: str = Field(default="", description="渲染图相对 data/ 的路径")

    vision_text: str = Field(default="", description="视觉通道产物，由 vision.py 填充")

    @property
    def cite_id(self) -> str:
        """本页作为证据被引用时的标记，如 P07。"""
        return f"P{self.page_no:02d}"

    @property
    def is_text_sparse(self) -> bool:
        """文字层是否近乎为空——这类页完全依赖视觉通道。"""
        from core.config import MIN_TEXT_CHARS

        return self.char_count < MIN_TEXT_CHARS

    def merged_text(self) -> str:
        """供索引与生成使用的合并文本。"""
        parts = [f"【{self.cite_id}】{self.title}".rstrip()]
        if self.text_layer:
            parts.append(self.text_layer)
        if self.vision_text:
            parts.append(f"[图像内容]\n{self.vision_text}")
        return "\n".join(parts)


class Chunk(BaseModel):
    """检索与引用的基本单元。

    通常一页一个 chunk；分节页因自身几无内容，并入其后一页，
    此时 chunk_id 形如 P08-09，引用时同时指向两页。
    """

    chunk_id: str = Field(description="引用标记，如 P12 或 P08-09")
    page_nos: list[int]
    section: str = Field(default="", description="所属章节，由分节页向后传播")
    title: str = ""
    text_layer: str = ""
    vision_text: str = ""
    image_paths: list[str] = Field(default_factory=list)
    retrievable: bool = Field(
        default=True,
        description="是否进入检索索引。目录页关键词覆盖广而无实质内容，置 False",
    )

    @property
    def char_count(self) -> int:
        return len(self.index_text())

    # ── retrieval.corpus.Indexable 协议 ──
    # 检索层不认识 Chunk，只认识「有 cite 和 heading 的东西」。
    # 知识库的锚点是页码，站点语料的锚点是 URL，二者在此归一。

    @property
    def cite(self) -> str:
        return self.chunk_id

    @property
    def heading(self) -> str:
        return f"《{self.title}》 · {self.section or '前言'}"

    def index_text(self) -> str:
        """送入向量库与 Prompt 的文本形态。

        章节名一并写入：既补充上下文提升召回，也让生成时能判断
        某条证据出自品宣稿的哪一部分。
        """
        parts = [f"【{self.chunk_id}】{self.title}".rstrip()]
        if self.section:
            parts.append(f"所属章节：{self.section}")
        if self.text_layer:
            parts.append(self.text_layer)
        if self.vision_text:
            parts.append(f"[图像内容]\n{self.vision_text}")
        return "\n".join(parts)
