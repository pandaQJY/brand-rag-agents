"""生成解析结果的可视化对照页，用于人工抽检转写质量。

pages.jsonl 是紧凑 JSON，中文被转义后无法肉眼阅读。本脚本把它渲染成
一张 HTML：左侧页面原图，右侧文字通道与视觉通道并排，便于逐页核对
视觉模型有没有编造内容。

用法：
    python -m ingest.preview      # 生成 data/preview.html
"""

from __future__ import annotations

import html
import re

from core.config import DATA_DIR
from kb.models import PageDoc
from kb.parse import load_pages

OUT = DATA_DIR / "preview.html"


def _render_body(text: str) -> str:
    """把转写里的 ## 小标题渲染出来，其余按段落。"""
    if not text.strip():
        return '<p class="empty">（空）</p>'
    out: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            out.append(f'<h4>{html.escape(s.lstrip("# ").strip())}</h4>')
        else:
            out.append(f"<p>{html.escape(s)}</p>")
    return "\n".join(out)


def _page_block(p: PageDoc) -> str:
    sparse = ' <span class="tag warn">文字层近空</span>' if p.is_text_sparse else ""
    gain = len(p.vision_text) - p.char_count
    gain_tag = f' <span class="tag ok">视觉 +{gain}</span>' if gain > 0 else ""
    return f"""
<section id="p{p.page_no}">
  <header>
    <span class="cite">{p.cite_id}</span>
    <h2>{html.escape(p.title or "（无标题）")}</h2>
    <div class="meta">
      文字层 {p.char_count} 字 · 视觉转写 {len(p.vision_text)} 字 · 图片 {p.image_count} 张{sparse}{gain_tag}
    </div>
  </header>
  <div class="grid">
    <figure class="shot">
      <img src="{p.image_path}" alt="{p.cite_id} 页面原图" loading="lazy">
      <figcaption>{p.image_path}</figcaption>
    </figure>
    <div class="col">
      <div class="colhead src-text">text_layer<small>PDF 文字层 · 原始字符 · 可信</small></div>
      <div class="content">{_render_body(p.text_layer)}</div>
    </div>
    <div class="col">
      <div class="colhead src-vision">vision_text<small>视觉模型转写 · 判读产物 · 需核查</small></div>
      <div class="content">{_render_body(p.vision_text)}</div>
    </div>
  </div>
</section>"""


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#FCFCFD;color:#0F1521;
  font:15px/1.7 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:0 24px 80px}
.top{padding:36px 0 20px;border-bottom:2px solid #0F1521;margin-bottom:8px}
h1{font-size:26px;margin:0 0 10px}
.stats{display:flex;flex-wrap:wrap;gap:0;font-size:13px;color:#4A5666}
.stats div{padding-right:22px;margin-right:22px;border-right:1px solid #DBE2EC}
.stats div:last-child{border:none}
.stats b{font-family:ui-monospace,Menlo,monospace;color:#1F44D6;font-size:16px}
.nav{position:sticky;top:0;background:#FCFCFDF2;backdrop-filter:blur(6px);
  padding:10px 0;border-bottom:1px solid #DBE2EC;z-index:10;
  display:flex;flex-wrap:wrap;gap:5px}
.nav a{font:11px ui-monospace,Menlo,monospace;color:#4A5666;text-decoration:none;
  padding:3px 7px;border:1px solid #DBE2EC;border-radius:3px}
.nav a:hover{border-color:#1F44D6;color:#1F44D6}
section{padding:34px 0;border-bottom:1px solid #DBE2EC;scroll-margin-top:56px}
header{margin-bottom:16px}
.cite{font:600 12px ui-monospace,Menlo,monospace;color:#1F44D6;
  background:#E7ECFC;padding:2px 8px;border-radius:3px}
h2{font-size:19px;margin:8px 0 6px}
.meta{font-size:12.5px;color:#77828F}
.tag{font:600 10.5px ui-monospace,Menlo,monospace;padding:1px 7px;border-radius:99px;margin-left:6px}
.tag.warn{background:#FBF0DA;color:#A26A05}
.tag.ok{background:#E1F2EA;color:#17795A}
.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr) minmax(0,1.25fr);gap:18px;align-items:start}
.shot{margin:0}
.shot img{width:100%;border:1px solid #C3CEDC;border-radius:5px;display:block}
.shot figcaption{font:11px ui-monospace,Menlo,monospace;color:#77828F;margin-top:6px}
.col{border:1px solid #DBE2EC;border-radius:5px;overflow:hidden;background:#fff}
.colhead{padding:9px 13px;font:600 12px ui-monospace,Menlo,monospace;
  border-bottom:1px solid #DBE2EC;display:flex;flex-direction:column;gap:2px}
.colhead small{font:400 10.5px -apple-system,sans-serif;opacity:.75}
.src-text{background:#F2F5F9;color:#4A5666}
.src-vision{background:#E7ECFC;color:#1F44D6}
.content{padding:12px 14px;max-height:520px;overflow-y:auto;font-size:13.5px}
.content p{margin:0 0 5px}
.content h4{margin:12px 0 5px;font-size:12px;color:#1F44D6;
  font-family:ui-monospace,Menlo,monospace;letter-spacing:.04em}
.content h4:first-child{margin-top:0}
.empty{color:#77828F;font-style:italic}
@media (prefers-color-scheme:dark){
  body{background:#0C1017;color:#E8ECF2}
  .top{border-color:#E8ECF2}
  .stats{color:#98A4B4}.stats div{border-color:#232C39}.stats b{color:#7B93FF}
  .nav{background:#0C1017F2;border-color:#232C39}
  .nav a{color:#98A4B4;border-color:#232C39}
  .nav a:hover{border-color:#7B93FF;color:#7B93FF}
  section{border-color:#232C39}
  .cite{background:#182140;color:#7B93FF}
  .meta,.shot figcaption,.empty{color:#6E7B8C}
  .tag.warn{background:#2A2114;color:#D9A23C}
  .tag.ok{background:#11291F;color:#3FBF8F}
  .shot img{border-color:#33404F}
  .col{border-color:#232C39;background:#141A24}
  .colhead{border-color:#232C39}
  .src-text{background:#1B2330;color:#98A4B4}
  .src-vision{background:#182140;color:#7B93FF}
  .content h4{color:#7B93FF}
}
@media (max-width:1100px){.grid{grid-template-columns:1fr}.content{max-height:none}}
"""


def build(pages: list[PageDoc]) -> str:
    tl = sum(p.char_count for p in pages)
    vt = sum(len(p.vision_text) for p in pages)
    nav = "".join(f'<a href="#p{p.page_no}">{p.cite_id}</a>' for p in pages)
    body = "".join(_page_block(p) for p in pages)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>解析结果对照 · {len(pages)} 页</title><style>{CSS}</style></head>
<body><div class="wrap">
<div class="top">
  <h1>解析结果对照表</h1>
  <div class="stats">
    <div>页数 <b>{len(pages)}</b></div>
    <div>文字层 <b>{tl:,}</b> 字</div>
    <div>视觉转写 <b>{vt:,}</b> 字</div>
    <div>视觉通道占比 <b>{vt / (tl + vt) * 100:.0f}%</b></div>
    <div>数据源 <b>pages.jsonl</b></div>
  </div>
</div>
<nav class="nav">{nav}</nav>
{body}
</div></body></html>"""


def main() -> None:
    pages = load_pages()
    OUT.write_text(build(pages), encoding="utf-8")
    print(f"已生成 {OUT}")
    print(f"打开：open {OUT}")


if __name__ == "__main__":
    main()
