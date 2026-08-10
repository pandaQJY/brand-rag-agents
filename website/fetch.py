"""官网抓取与结构化解析。

与知识库链路的 PDF 解析有一处本质差异：那里只关心"说了什么"，
这里还必须关心"怎么组织的"——因为 GEO 诊断的核心判据正是结构信号：
有没有 h1、标题层级是否合理、有没有 JSON-LD、meta 描述是否完整、
有没有 FAQ 式问答结构。因此解析保留结构，不剥成纯文本。

实测该站为服务端渲染的静态 HTML（无 __NEXT_DATA__ / #app / #root），
故用 httpx + BeautifulSoup 即可，无需 Playwright。

用法：
    python -m website.fetch              # 抓取全站
    python -m website.fetch --max 5      # 只抓 5 页（调试）
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from core.config import (
    BOILERPLATE_JSON,
    CRAWL_DELAY,
    CRAWL_MAX_PAGES,
    SITE_BASE,
    SITE_PAGES_JSONL,
    site_domain,
)
from website.models import Heading, SitePage

BASE = SITE_BASE
DOMAIN = site_domain()  # 站内链接判据，随 SITE_BASE 自动变化
PAGES_JSONL = SITE_PAGES_JSONL

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 法务页无 GEO 分析价值，排除
SKIP = ("/privacy", "/terms")
ASSET = (".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp", ".pdf", ".mp4")

_WS = re.compile(r"\s+")

# 正文按文档顺序抽取的块级标签。**标题必须在内**——这不是格式偏好，是硬需求。
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li"]


def _clean(t: str) -> str:
    return _WS.sub(" ", t).strip()


def _blocks(body) -> list[str]:
    """按文档顺序抽出块级文本，标题带 Markdown 层级标记。

    初版正文只收 p 与 li，标题标签的文字一个字都不进正文。后果在首页 FAQ 上
    暴露得最清楚：问句写在 <h3> 里、答案写在 <p> 里，于是模型看到的是**一串
    没有问题的答案**——孤立读来像营销段落，读不出这是一组问答。据此得出的
    「官网缺少 GEO 与 SEO 区别的 FAQ」是错的，那条 FAQ 就在首页上。

    实测全站 402 条标题里只有 126 条侥幸可见（恰好在别处以 p/li 重复出现）。
    保留层级标记则更进一步：模型不只看得见标题，还看得出谁统辖谁。
    """
    out: list[str] = []
    for el in body.find_all(BLOCK_TAGS):
        # <li> 内嵌 <p> 时两者都会被命中，只取最内层，否则同一句会重复两遍
        if el.find(BLOCK_TAGS):
            continue
        text = _clean(el.get_text())
        is_head = el.name[0] == "h"
        if len(text) < (2 if is_head else 9):
            continue
        out.append(f"{'#' * int(el.name[1])} {text}" if is_head else text)
    return out


def _normalize(url: str) -> str:
    p = urlparse(urljoin(BASE, url))
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/') or '/'}"


def _internal(url: str) -> bool:
    p = urlparse(url)
    return (
        p.netloc.endswith(DOMAIN)
        and not any(p.path.lower().endswith(e) for e in ASSET)
        and not any(s in p.path for s in SKIP)
    )


def parse_page(url: str, html: str) -> SitePage:
    """解析一页，同时抽取内容与结构信号。"""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        if tag.name == "script" and tag.get("type") == "application/ld+json":
            continue
        tag.decompose()

    title = _clean(soup.title.get_text()) if soup.title else ""
    meta_desc = ""
    if (m := soup.find("meta", attrs={"name": "description"})):
        meta_desc = _clean(m.get("content", ""))
    if not meta_desc and (m := soup.find("meta", attrs={"property": "og:description"})):
        meta_desc = _clean(m.get("content", ""))

    headings = [
        Heading(level=int(h.name[1]), text=_clean(h.get_text()))
        for h in soup.find_all(["h1", "h2", "h3", "h4"])
        if _clean(h.get_text())
    ]

    # JSON-LD 结构化数据——AI 抓取的重要信号
    jsonld: list[str] = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = s.string or s.get_text()
        if raw and raw.strip():
            try:
                obj = json.loads(raw)
                jsonld.append(obj.get("@type", "未标注类型") if isinstance(obj, dict) else "数组")
            except json.JSONDecodeError:
                jsonld.append("解析失败")

    body = soup.body or soup
    text = "\n".join(_blocks(body)) or _clean(body.get_text())

    links = sorted({_normalize(a["href"]) for a in soup.find_all("a", href=True)
                    if _internal(_normalize(a["href"]))})

    return SitePage(
        url=url,
        title=title,
        meta_description=meta_desc,
        headings=headings,
        text=text,
        char_count=len(_WS.sub("", text)),
        jsonld_types=jsonld,
        image_count=len(soup.find_all("img")),
        images_without_alt=sum(1 for i in soup.find_all("img") if not i.get("alt", "").strip()),
        internal_links=links,
    )


def crawl(max_pages: int = CRAWL_MAX_PAGES, delay: float = CRAWL_DELAY) -> list[SitePage]:
    """广度优先抓取全站。站点很小，无需并发。"""
    seen: set[str] = set()
    queue = [_normalize(BASE)]
    pages: list[SitePage] = []

    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=20) as client:
        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            try:
                r = client.get(url)
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {url}  {type(e).__name__}")
                continue

            page = parse_page(url, r.text)
            pages.append(page)
            print(f"  ✓ {page.char_count:>5}字  h1×{page.h1_count}  {url}")

            for link in page.internal_links:
                if link not in seen:
                    queue.append(link)
            time.sleep(delay)

    return pages


def strip_boilerplate(pages: list[SitePage], threshold: float = 0.5) -> list[str]:
    """从各页正文中剔除跨页重复的模板文案，返回被剔除的行，并重算字数。

    每页正文都以四条 🎉 公告条开头、以电话邮箱备案号结尾，合计 200 余字。
    对 /geo-center/whitepapers 这种全页 382 字的页面，模板占了一半以上——
    送进模型的样本里，真正属于这一页的内容反而是少数。

    判据沿用 scan.check_faq_structure 已验证的那条：**出现在超过半数页面的行是
    模板，不是内容。** 同一条主张在抓取层与诊断层各用一次，不必维护两套逻辑。

    剔除而不丢弃：模板里有 slogan 和公告，那是品牌表达的一部分，
    因此单独返回、单独存档，供 Agent 通读一次。
    """
    freq = Counter(line for p in pages for line in set(p.text.splitlines()))
    boiler = {ln for ln, n in freq.items() if n > len(pages) * threshold}
    for p in pages:
        p.text = "\n".join(ln for ln in p.text.splitlines() if ln not in boiler)
        p.char_count = len(_WS.sub("", p.text))
    # 按跨页出现次数排序，最通用的排前面
    return sorted(boiler, key=lambda ln: -freq[ln])


def save(pages: list[SitePage], path=PAGES_JSONL) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(p.model_dump_json() + "\n")


def load(path=PAGES_JSONL) -> list[SitePage]:
    with open(path, encoding="utf-8") as f:
        return [SitePage.model_validate_json(ln) for ln in f if ln.strip()]


def load_boilerplate(path=BOILERPLATE_JSON) -> list[str]:
    """读取全站模板文案。未抓取过则返回空列表，不影响运行。"""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=f"抓取站点 {BASE}")
    ap.add_argument("--max", type=int, default=CRAWL_MAX_PAGES)
    args = ap.parse_args()

    print(f"抓取 {BASE} …")
    pages = crawl(args.max)

    raw_total = sum(p.char_count for p in pages)
    boiler = strip_boilerplate(pages)
    BOILERPLATE_JSON.write_text(
        json.dumps(boiler, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save(pages)

    total = sum(p.char_count for p in pages)
    print(f"\n剔除全站模板 {len(boiler)} 行，正文由 {raw_total:,} 字降至 {total:,} 字 "
          f"→ {BOILERPLATE_JSON.name}")
    no_h1 = [p for p in pages if p.h1_count == 0]
    no_meta = [p for p in pages if not p.meta_description]
    no_ld = [p for p in pages if not p.jsonld_types]

    print(f"\n完成：{len(pages)} 页 / 共 {total:,} 字 → {PAGES_JSONL}")
    print(f"  缺 h1        {len(no_h1):>2}/{len(pages)} 页")
    print(f"  缺 meta 描述  {len(no_meta):>2}/{len(pages)} 页")
    print(f"  无 JSON-LD   {len(no_ld):>2}/{len(pages)} 页")
    print(f"  图片缺 alt    {sum(p.images_without_alt for p in pages)}/"
          f"{sum(p.image_count for p in pages)} 张")


if __name__ == "__main__":
    main()
