"""把官网数据压成 Agent 可读的上下文。

全站 9.6 万字塞不进 Prompt，也没必要。关键是**丢什么、怎么丢**，本模块的三条规则：

    ① 覆盖靠大纲，不靠抽样    全站 h1–h3 目录整体给出，每一页都露面
    ② 截断按结构块，且留痕    切在小节边界，并写明略去了哪几节
    ③ 一个总预算，按占比分配  换站点不必逐项调参

## ① 为什么覆盖必须由大纲保证

初版给 SiteAnalyst 的是「四个身份页正文 + 其余页标题」，全站 105,042 字里
只有 4.4% 被读到。改成按板块抽样正文后覆盖到 8 页，仍然是抽样——
**抽样保证不了「有没有」类问题的答案**。实测翻车：官网首页有一整块 FAQ，
第一条就是「GEO 到底是什么？和 SEO 有什么本质区别？」，而系统判出
「官网缺少 GEO 与 SEO 区别的 FAQ」。

正确的解法不是把样本调大，是换个东西回答这类问题：**站点自己写的标题目录**。
全站 421 条 h1–h3 渲染成大纲约 18k 字，比只覆盖 8 页正文的样本还便宜，
却让 52 页全部露面。正文样本就此退居辅助——它只需回答「这个站怎么说话」，
那本来就只要几篇代表作。

## ② 为什么截断必须留痕

首页 FAQ 起始于正文第 3,016 字，而单页额度是 3,000。差 16 个字，整块被截掉，
而且**截得悄无声息**——模型不知道自己没看到，于是理直气壮地判「缺少」。
丢失不可避免，丢得没有痕迹才是 bug。现在正文带 `#` 层级，可以切在小节边界，
并留下「本页另有 N 个小节未展开：A、B、C」。

## ③ 为什么只留一个绝对数

初版三个绝对字数常数都是照着这个站试出来的，换站即失效。现在只有
config.SITE_DIGEST_BUDGET 一个总额，其余是占比：页多则每页自动收紧，
页少则全给。
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from core.config import (
    IDENTITY_SHARE,
    OUTLINE_SHARE,
    SECTION_SAMPLES,
    SITE_DIGEST_BUDGET,
)
from website.fetch import load, load_boilerplate

# 这些页面承担品牌身份说明，正文优先完整给出
IDENTITY_PATHS = ("/", "/about", "/geo-agent", "/contact")

_norm = lambda p: p.rstrip("/") or "/"  # noqa: E731
_IDENTITY = {_norm(p) for p in IDENTITY_PATHS}

# 分节标记。不能再用 Markdown 的 # 号——自 fetch.py 把页面标题收进正文后，
# 正文里本来就有 ## / ###，两者混在一起模型分不清哪个是页面边界、哪个是页内结构。
_MARK = "▮"

_HEAD_LINE = re.compile(r"^(#{1,6})\s+(.*)$")

# 每段正文除自身外还要占两行：页面标记，以及 _fit 可能追加的「另有 N 个小节未展开」。
# 分配额度时先扣掉，否则预算按正文算、实际按渲染结果算，必然超支。
_OVERHEAD = 140


# ── 页面工具 ────────────────────────────────────────────────

def _unique(pages: list) -> list:
    """按路径去重——同一内容可能被多个 URL 抓到，见 scan.check_url_canonical()。"""
    seen: set[str] = set()
    out = []
    for p in pages:
        if _norm(p.path) not in seen:
            seen.add(_norm(p.path))
            out.append(p)
    return out


def _section(page) -> str:
    """页面所属板块，取路径首段：/geo-center/reports/xxx → /geo-center。"""
    head = _norm(page.path).lstrip("/").split("/")[0]
    return f"/{head}" if head else "/"


def _subgroup(page) -> str:
    """板块内的子目录：/geo-center/reports/xxx → reports；/blog/xxx → ""（无子目录层）。"""
    parts = [s for s in page.path.split("/") if s]
    return parts[1] if len(parts) > 2 else ""


def _round_robin(pages: list, n: int) -> list:
    """在板块内按子目录轮转取样，每轮取该子目录里最长的一篇。

    子目录按页数从多到少出场：主体子目录先保证覆盖，小子目录也至少出一篇。
    不轮转的话，/geo-center 下 13 篇行业报告会把白皮书与视频课全部挤掉，
    而后两者恰恰写着产品方法论。
    """
    buckets: dict[str, list] = {}
    for p in pages:
        buckets.setdefault(_subgroup(p), []).append(p)
    for v in buckets.values():
        v.sort(key=lambda p: -p.char_count)
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), k))

    out: list = []
    while len(out) < n and any(buckets.values()):
        for key in order:
            if buckets[key] and len(out) < n:
                out.append(buckets[key].pop(0))
    return out


def _show_path(page) -> str:
    """展示用路径：把百分号编码解回中文。

    站内报告页的 slug 是中文标题直接编码的，原样喂进去是近 2,700 个字符的
    `%E6%9D%83` 噪声——既占额度又不表达任何信息。
    """
    return unquote(page.path)


def _title_of(page) -> str:
    return next((h.text for h in page.headings if h.level == 1), "") or page.title


def _page_head(page, limit: int = 50) -> str:
    return f"{_MARK} 页面 {_show_path(page)}　{_title_of(page)[:limit]}　{page.char_count}字"


# ── ② 按结构块截断 ──────────────────────────────────────────

def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 `#` 标题把正文切成 [(小节名, 含标题行的整段)]。标题前的内容归入「开头」。"""
    out: list[tuple[str, list[str]]] = [("开头", [])]
    for line in text.splitlines():
        if m := _HEAD_LINE.match(line):
            out.append((m.group(2), [line]))
        else:
            out[-1][1].append(line)
    return [(name, "\n".join(body).strip()) for name, body in out if "\n".join(body).strip()]


def _fit(text: str, budget: int) -> str:
    """把正文裁到预算内，切在小节边界，并写明略去了哪几节。

    绝不从第 N 个字符一刀砍下——那样既切碎句子，又让丢失完全静默。
    """
    if len(text) <= budget:
        return text

    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    for name, body in _split_sections(text):
        if used + len(body) + 1 <= budget:
            kept.append(body)
            used += len(body) + 1
        else:
            dropped.append(name)

    if not kept:  # 单节就超预算，只能硬截，但仍然说明白
        return f"{text[:budget]}…\n（本页正文共 {len(text)} 字，此处截断）"

    tail = "、".join(dropped[:8]) + ("等" if len(dropped) > 8 else "")
    return "\n".join(kept) + f"\n（本页另有 {len(dropped)} 个小节未展开：{tail}）"


def _allocate(pages: list, budget: int, floor: int = 400) -> dict[str, int]:
    """按正文长度比例分额度，每页保底 floor 字——长页多给，但不许吃光。

    额度是给正文的，渲染开销（页面标记行、留痕说明）先从总额里扣掉。
    """
    net = max(0, budget - _OVERHEAD * len(pages))
    total = sum(p.char_count for p in pages) or 1
    return {p.url: max(floor, int(net * p.char_count / total)) for p in pages}


# ── ① 全站标题大纲 ──────────────────────────────────────────

def outline_budget() -> int:
    """大纲的默认额度。凡是要对全站下断言的环节都用它，口径统一。"""
    return int(SITE_DIGEST_BUDGET * OUTLINE_SHARE)


def site_outline(budget: int | None = None) -> str:
    """全站标题大纲——站点自己写的目录，覆盖每一页。

    预算不够时按这个梯子降级，而不是砍掉后面的页面：
        标题截到 60 字 → 截到 30 字 → 只留 h1–h2 → 只留 h1
    **少看几层结构，每页仍然看得见；砍页面则整页凭空消失。**

    降级时**问句式标题一律保留，不论它在第几层**。这不是为这个站开的后门：
    对 GEO 分析而言问句标题是价值最高的结构信号，它直接对应「客户会问什么、
    官网答没答」。首页那组 FAQ 恰好写在 h3 上，按层级一刀切，头一个丢的就是它。

    降级同样要留痕——顶部写明本次列到第几层，模型才知道自己读的是删节版。
    """
    budget = budget or outline_budget()
    pages = _unique(load())
    boiler = {ln.lstrip("# ").strip() for ln in load_boilerplate()}

    def render(cap: int, max_level: int) -> str:
        lines: list[str] = []
        for p in pages:
            heads = [
                h for h in p.headings
                if h.text not in boiler
                and (h.level <= max_level or h.text.rstrip().endswith(("？", "?")))
            ]
            lines.append(f"{_MARK} 页面 {_show_path(p)}　{p.char_count}字")
            lines += ["  " * (h.level - 1) + f"h{h.level} {h.text[:cap]}" for h in heads] or [
                "  （无标题标签）"
            ]
        return "\n".join(lines)

    for cap, max_level in ((60, 3), (30, 3), (30, 2), (30, 1)):
        out = render(cap, max_level)
        if len(out) <= budget:
            if (cap, max_level) == (60, 3):
                return out
            trim = f"，标题截至 {cap} 字" if cap < 60 else ""
            return (f"（篇幅所限，本大纲只列到 h{max_level}{trim}；"
                    f"问句式标题不受此限，已全部保留）\n{out}")
    # 梯子走到底仍然超预算：到这一步覆盖已经保不住，只能少列页面。
    # 那就按整页切，并写明少了哪几页——绝不从半行字中间截断，
    # 那会让模型以为最后那页只有半个标题，而不是根本没列全。
    blocks = re.split(rf"(?=^{_MARK} 页面 )", out, flags=re.M)
    kept: list[str] = []
    used = 0
    for b in blocks:
        if used + len(b) > budget:
            break
        kept.append(b)
        used += len(b)
    missing = len(pages) - max(0, len(kept) - 1)
    return (f"（预算过紧，本大纲只列到 h1，且仅收录 {len(pages) - missing}/{len(pages)} 页；"
            f"其余 {missing} 页未列出）\n" + "".join(kept))


# ── 主入口 ──────────────────────────────────────────────────

def site_digest(budget: int = SITE_DIGEST_BUDGET) -> str:
    """全站概览：大纲覆盖每一页，身份页与板块样本补充正文。"""
    pages = _unique(load())
    lines = [
        f"官网共 {len(pages)} 页，合计 {sum(p.char_count for p in pages):,} 字。",
        f"以 {_MARK} 开头的行是本清单的分节标记；正文里的 # 号标题是页面自身的结构。",
        "",
        f"{_MARK}{_MARK} 一、全站标题大纲（{len(pages)} 页全部，这是官网自己的目录）",
        site_outline(int(budget * OUTLINE_SHARE)),
        "",
    ]

    # 模板文案已在抓取时从各页正文剔除（见 fetch.strip_boilerplate），此处补回一次。
    # 它是 slogan、公告与全局导航，属于品牌表达，该被读到——只是不该读 52 遍。
    if boiler := load_boilerplate():
        lines.append(f"{_MARK}{_MARK} 二、全站模板文案（每页重复出现，此处只列一次）")
        lines += [f"- {ln}" for ln in boiler]
        lines.append("")

    ident = [p for p in pages if _norm(p.path) in _IDENTITY]
    quota = _allocate(ident, int(budget * IDENTITY_SHARE))
    lines.append(f"{_MARK}{_MARK} 三、品牌身份页正文")
    for p in ident:
        lines += [_page_head(p, 40), _fit(p.text, quota[p.url]), ""]

    rest = [p for p in pages if _norm(p.path) not in _IDENTITY]
    by_section: dict[str, list] = {}
    for p in rest:
        by_section.setdefault(_section(p), []).append(p)

    lines.append(f"{_MARK}{_MARK} 四、各板块正文样本（判断表达方式用，覆盖见上面的大纲）")
    left = max(0, budget - len("\n".join(lines)))
    for sec in sorted(by_section, key=lambda s: (-len(by_section[s]), s)):
        group = by_section[sec]
        share = left // len(by_section)
        # 预算紧时减少样本篇数，而不是让每篇都缩到读不出东西——
        # 保底 400 字以下的样本判断不了表达方式，给了等于没给，还顶穿预算。
        picks = _round_robin(group, min(SECTION_SAMPLES, max(1, share // (400 + _OVERHEAD))))
        quota = _allocate(picks, share)
        lines.append(f"{_MARK} 板块 {sec}（共 {len(group)} 页，以下 {len(picks)} 篇为样本）")
        for p in picks:
            lines += [_page_head(p), _fit(p.text, quota[p.url]), ""]

    return "\n".join(lines)


def sample_bodies(n: int = 6, budget: int | None = None, prefix: str = "") -> str:
    """跨板块的正文样本，供判断表达方式时使用。

    prefix 留空表示不限板块。留空是默认值而非可选项——判断「产品与服务页
    是否具备问答式结构」，样本里却一篇产品页都没有，这个判断就不成立。
    """
    budget = budget or int(SITE_DIGEST_BUDGET * (1 - OUTLINE_SHARE - IDENTITY_SHARE))
    pages = [p for p in _unique(load()) if _norm(p.path) not in _IDENTITY]
    if prefix:
        pages = [p for p in pages if p.path.startswith(prefix)]

    by_section: dict[str, list] = {}
    for p in pages:
        by_section.setdefault(_section(p), []).append(p)

    # 板块之间同样轮转，保证 n 篇样本不会全来自同一板块
    queues = {s: _round_robin(v, n) for s, v in by_section.items()}
    order = sorted(queues, key=lambda s: (-len(by_section[s]), s))
    picked: list = []
    while len(picked) < n and any(queues.values()):
        for s in order:
            if queues[s] and len(picked) < n:
                picked.append(queues[s].pop(0))

    quota = _allocate(picked, budget)
    return "\n\n".join(f"{_page_head(p)}\n{_fit(p.text, quota[p.url])}" for p in picked)
