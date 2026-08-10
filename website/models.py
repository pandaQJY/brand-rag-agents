"""官网页面的数据结构。

字段分两类：**内容字段**（title / text）与**结构信号字段**
（headings / jsonld_types / meta_description / images_without_alt）。
后者是 GEO 诊断的直接依据——AI 抓取时靠它们理解页面组织与主题，
因此必须与正文一并保留，不能剥成纯文本。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Heading(BaseModel):
    level: int = Field(description="1-4，对应 h1-h4")
    text: str


class SitePage(BaseModel):
    url: str
    title: str = ""
    meta_description: str = ""
    headings: list[Heading] = Field(default_factory=list)
    text: str = ""
    char_count: int = 0

    # ── 结构信号 ──
    jsonld_types: list[str] = Field(default_factory=list, description="JSON-LD 的 @type 列表")
    image_count: int = 0
    images_without_alt: int = 0
    internal_links: list[str] = Field(default_factory=list)

    @property
    def path(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self.url).path or "/"

    @property
    def h1_count(self) -> int:
        return sum(1 for h in self.headings if h.level == 1)

    @property
    def heading_outline(self) -> str:
        """标题层级大纲，供 Agent 判断信息组织是否清晰。"""
        return "\n".join("  " * (h.level - 1) + f"h{h.level} {h.text}" for h in self.headings)

    def summary_line(self) -> str:
        return f"{self.path}\t{self.title[:40]}\t{self.char_count}字\th1×{self.h1_count}"

    def for_agent(self, max_chars: int = 3000) -> str:
        """喂给 Agent 的形态：结构信号在前，正文在后。"""
        parts = [
            f"URL: {self.url}",
            f"标题: {self.title or '（无）'}",
            f"meta描述: {self.meta_description or '（缺失）'}",
            f"JSON-LD: {', '.join(self.jsonld_types) or '（无结构化标记）'}",
            f"标题层级: h1×{self.h1_count}",
            self.heading_outline or "（无标题标签）",
            "--- 正文 ---",
            self.text[:max_chars],
        ]
        return "\n".join(parts)
