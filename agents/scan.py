"""确定性结构扫描：不调用任何模型，纯代码产出 B 档诊断结论。

GEO 诊断的一部分判据是**可观测事实**——title 是否唯一、有无 JSON-LD、
h1 数量、alt 缺失率。把这些交给 LLM 判断既慢又不稳定（写作链路实测过
核查器的采样波动：同一输入两次结论不同）。凡能确定性裁决的，就不交给模型。

产出的每条 Finding 都带证据档位：
    B —— 本模块产出的可观测事实，结果可复现
    A —— 与品宣稿自述标准冲突（由 standard.py 补充判定）

用法：
    python -m agents.scan
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

from core.config import SCAN_JSON
from website.fetch import load


@dataclass
class Finding:
    """一条可观测的结构发现。"""

    key: str
    title: str
    grade: str  # A | B | C
    severity: str  # high | medium | low | ok
    fact: str  # 客观事实陈述，带数字
    detail: str = ""  # 补充说明
    pages: list[str] = field(default_factory=list)  # 相关页面路径

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "grade": self.grade,
            "severity": self.severity,
            "fact": self.fact,
            "detail": self.detail,
            "pages": self.pages[:10],
        }


# ── 各项检查 ────────────────────────────────────────────────

def check_title_uniqueness(pages) -> Finding:
    titles = Counter(p.title for p in pages if p.title)
    uniq = len(titles)
    total = len(pages)
    if uniq == 1 and total > 1:
        top = titles.most_common(1)[0][0]
        return Finding(
            key="title_uniqueness",
            title="title 标签全站同质",
            grade="B",
            severity="high",
            fact=f"{total} 个页面的 <title> 去重后仅 {uniq} 种",
            detail=f"全部为「{top[:40]}」。搜索结果列表中无法区分，"
                   f"部分抓取管线会据此判定重复内容。注意这是元数据层问题，"
                   f"页面内部标题结构另行评估。",
        )
    ratio = uniq / total if total else 0
    return Finding(
        key="title_uniqueness",
        title="title 标签唯一性",
        grade="B",
        severity="ok" if ratio > 0.8 else "medium",
        fact=f"{total} 页共 {uniq} 种 title（唯一率 {ratio:.0%}）",
    )


def check_meta_uniqueness(pages) -> Finding:
    metas = Counter(p.meta_description for p in pages if p.meta_description)
    uniq = len(metas)
    missing = [p.path for p in pages if not p.meta_description]
    if uniq <= 1 and len(pages) > 1:
        return Finding(
            key="meta_uniqueness",
            title="meta description 全站同质",
            grade="B",
            severity="high",
            fact=f"{len(pages)} 个页面的 meta description 去重后仅 {uniq} 种",
            detail="页面摘要无差异，AI 抓取时无法据此区分内容主题。",
        )
    return Finding(
        key="meta_uniqueness",
        title="meta description 覆盖与唯一性",
        grade="B",
        severity="ok" if not missing else "medium",
        fact=f"{len(pages) - len(missing)}/{len(pages)} 页有 meta，去重 {uniq} 种",
        pages=missing,
    )


def check_jsonld(pages) -> Finding:
    with_ld = [p for p in pages if p.jsonld_types]
    n = len(with_ld)
    if n == 0:
        return Finding(
            key="jsonld",
            title="全站无 JSON-LD 结构化标记",
            grade="B",
            severity="high",
            fact=f"{len(pages)} 个页面中 JSON-LD 覆盖数为 0",
            detail="结构化标记的作用在检索层——实体消歧、知识图谱收录、"
                   "问答对抽取。其对 AI 检索管线的加权作用无公开确证，"
                   "但与品宣稿自述标准的冲突可由 standard.py 判定为 A 档。",
        )
    return Finding(
        key="jsonld",
        title="JSON-LD 覆盖率",
        grade="B",
        severity="ok" if n == len(pages) else "medium",
        fact=f"{n}/{len(pages)} 页含 JSON-LD",
        pages=[p.path for p in pages if not p.jsonld_types],
    )


def check_heading_structure(pages) -> list[Finding]:
    """标题结构分两条：h1 数量异常，以及内容层是否健全。

    拆成两条是有意的——初版方案曾因只看 title 就断言「AI 读不懂页面」，
    核对 h1/h2/h3 后才发现内容层结构完好。诊断必须把元数据层与内容层分开说。
    """
    bad = [p for p in pages if p.h1_count != 1]
    h1_texts = [h.text for p in pages for h in p.headings if h.level == 1]
    total_headings = sum(len(p.headings) for p in pages)
    levels = Counter(h.level for p in pages for h in p.headings)

    out = [
        Finding(
            key="h1_anomaly",
            title="h1 数量异常的页面",
            grade="B",
            severity="medium" if bad else "ok",
            fact=f"{len(bad)}/{len(pages)} 页的 h1 数量不为 1",
            detail="；".join(f"{p.path} h1×{p.h1_count}" for p in bad[:5]) or "全部正常",
            pages=[p.path for p in bad],
        ),
        Finding(
            key="heading_health",
            title="内容层标题结构",
            grade="B",
            severity="ok",
            fact=f"全站 {total_headings} 个标题标签，h1 去重 "
                 f"{len(set(h1_texts))}/{len(h1_texts)}",
            detail="层级分布：" + " / ".join(f"h{k}×{v}" for k, v in sorted(levels.items()))
                   + "。内容层结构健全，与元数据层同质问题应分开评估。",
        ),
    ]
    return out


def check_image_alt(pages) -> Finding:
    total = sum(p.image_count for p in pages)
    missing = sum(p.images_without_alt for p in pages)
    ratio = missing / total if total else 0
    return Finding(
        key="image_alt",
        title="图片 alt 文本缺失",
        grade="C",  # 对 AI 理解的实际影响无确证，标为推测档
        severity="low" if ratio < 0.2 else "medium",
        fact=f"{missing}/{total} 张图片缺少 alt（{ratio:.0%}）",
        detail="alt 对大模型理解图像内容的实际影响缺乏公开确证，"
               "此项列为 C 档建议而非确定缺陷。",
        pages=[p.path for p in pages if p.images_without_alt],
    )


def check_url_canonical(pages) -> Finding:
    """同一内容是否存在多个可达 URL。"""
    by_content: dict[int, list[str]] = {}
    for p in pages:
        by_content.setdefault(p.char_count, []).append(p.url)
    dupes = {
        k: v for k, v in by_content.items()
        if len(v) > 1 and len({urlparse(u).path.rstrip("/") for u in v}) == 1
    }
    n = sum(len(v) - 1 for v in dupes.values())
    return Finding(
        key="url_canonical",
        title="URL 规范化",
        grade="B",
        severity="medium" if n else "ok",
        fact=f"检出 {n} 组同内容多 URL" if n else "未检出同内容多 URL",
        detail="；".join(" ↔ ".join(v) for v in list(dupes.values())[:3]),
    )


def check_content_depth(pages) -> Finding:
    """身份信息厚度——定义「我是谁、我能做什么」的页面是否足够充实。"""
    IDENTITY = ("/about", "/contact", "/geo-agent", "/")
    core = [p for p in pages if p.path.rstrip("/") in
            [x.rstrip("/") for x in IDENTITY] and p.path != "/"]
    blog = [p for p in pages if p.path.startswith("/blog/")]
    if not core or not blog:
        return Finding("content_depth", "身份信息厚度", "B", "ok", "样本不足，跳过")

    core_avg = sum(p.char_count for p in core) // len(core)
    blog_avg = sum(p.char_count for p in blog) // len(blog)
    inverted = core_avg < blog_avg
    return Finding(
        key="content_depth",
        title="身份信息厚度倒挂" if inverted else "身份信息厚度",
        grade="B",
        severity="medium" if inverted else "ok",
        fact=f"身份页均 {core_avg} 字（{len(core)} 页），blog 均 {blog_avg} 字（{len(blog)} 篇）",
        detail="当用户问 AI「这家公司是做什么的」，官方可提供的素材反而最少。"
               if inverted else "",
        pages=[f"{p.path}（{p.char_count}字）" for p in sorted(core, key=lambda x: x.char_count)],
    )


# ── 汇总 ────────────────────────────────────────────────────

def check_faq_structure(pages) -> Finding:
    """问答式结构——但必须剔除全站模板重复的问句。

    实测教训：全站 107 个问句标题分布在 51/51 页，看似覆盖完美；
    去重后才发现其中 52 个是同一句营销标语（出现在每页页脚）。
    若不去重，会得出「全站具备问答式结构」这一完全错误的结论。
    只有**页面独有**的问句才构成真正的问答式内容。
    """
    from collections import Counter

    q_all = [
        (p.path, h.text.strip())
        for p in pages
        for h in p.headings
        if h.text.rstrip().endswith(("？", "?"))
    ]
    freq = Counter(t for _, t in q_all)
    # 出现在超过半数页面的，判定为模板文案而非页面内容
    boiler = {t for t, n in freq.items() if n > len(pages) * 0.5}
    unique_q = [(path, t) for path, t in q_all if t not in boiler]
    pages_with_q = {path for path, _ in unique_q}

    ratio = len(pages_with_q) / len(pages) if pages else 0
    return Finding(
        key="faq_structure",
        title="问答式结构覆盖不足" if ratio < 0.5 else "问答式结构",
        grade="B",
        severity="medium" if ratio < 0.5 else "ok",
        fact=f"{len(pages_with_q)}/{len(pages)} 页含页面独有的问句式标题"
             f"（共 {len(unique_q)} 个）",
        detail=f"另有 {len(boiler)} 句为全站模板重复（如出现 {max(freq.values())} 次的"
               f"营销标语），已剔除不计。问答式结构集中在 blog，"
               f"而产品与服务页缺失——客户最可能向 AI 提问的正是后者。",
        pages=sorted(pages_with_q)[:10],
    )


def check_internal_link(pages) -> Finding:
    """内链结构——导航链接不算，只看内容页之间的实质关联。

    实测教训：按「零入链孤岛页」检查会得出 0 问题，因为全站导航使
    每个页面都有 51 条入链。导航链接对实体关联毫无贡献，必须排除。
    """
    from urllib.parse import urlparse

    paths = {p.path.rstrip("/") or "/" for p in pages}
    # 出现在几乎所有页面的链接目标 = 全局导航
    from collections import Counter

    target_freq = Counter()
    for p in pages:
        for l in set(p.internal_links):
            t = urlparse(l).path.rstrip("/") or "/"
            if t in paths:
                target_freq[t] += 1
    nav = {t for t, n in target_freq.items() if n > len(pages) * 0.8}

    content = [p for p in pages if (p.path.rstrip("/") or "/") not in nav]
    cross = 0
    isolated = []
    for p in content:
        src = p.path.rstrip("/") or "/"
        n = sum(
            1
            for l in set(p.internal_links)
            if (t := urlparse(l).path.rstrip("/") or "/") in paths
            and t not in nav
            and t != src
        )
        cross += n
        if n == 0:
            isolated.append(p.path)

    avg = cross / len(content) if content else 0
    return Finding(
        key="internal_link",
        title="内容页之间缺乏实质内链" if avg < 2 else "内链结构",
        grade="B",
        severity="medium" if avg < 2 else "ok",
        fact=f"{len(content)} 个内容页之间仅 {cross} 条互链（页均 {avg:.1f} 条），"
             f"其中 {len(isolated)} 页无任何内容页出链",
        detail=f"已排除 {len(nav)} 个全局导航目标——导航链接对实体关联无贡献。"
               f"内容页互不引用，AI 难以建立产品、案例、FAQ 之间的语义关系。",
        pages=isolated[:10],
    )


def check_knowledge_graph(pages) -> Finding:
    """知识图谱的可观测代理指标：实体页是否被内容页引用。

    真正的知识图谱无法从 HTML 直接测出。这里退而测一个必要条件——
    案例、产品、FAQ 等实体页若从未被正文引用，就不存在语义关系网络。
    """
    from urllib.parse import urlparse

    ENTITY = ("/about", "/geo-agent", "/contact")
    paths = {p.path.rstrip("/") or "/" for p in pages}
    body_refs = {e: 0 for e in ENTITY}

    # 仅统计 blog / news 正文页对实体页的引用（排除导航）
    for p in pages:
        if not (p.path.startswith("/blog/") or p.path.startswith("/news/")):
            continue
        for l in set(p.internal_links):
            t = urlparse(l).path.rstrip("/") or "/"
            if t in body_refs:
                body_refs[t] += 1

    linked = sum(1 for v in body_refs.values() if v > 0)
    return Finding(
        key="knowledge_graph",
        title="实体页未被内容引用" if linked == 0 else "实体关联",
        grade="B",
        severity="medium" if linked < len(ENTITY) else "ok",
        fact=f"{linked}/{len(ENTITY)} 个实体页被 blog/news 正文引用",
        detail="注：真正的知识图谱无法从 HTML 直接测出，此处测的是必要条件——"
               "若案例与产品页从未被正文引用，则不存在实体语义关系网络。"
               + "".join(f" {k}：{v} 次；" for k, v in body_refs.items()),
    )


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "ok": 3}


def scan(pages=None) -> list[Finding]:
    pages = pages if pages is not None else load()
    findings: list[Finding] = [
        check_title_uniqueness(pages),
        check_meta_uniqueness(pages),
        check_jsonld(pages),
        *check_heading_structure(pages),
        check_image_alt(pages),
        check_url_canonical(pages),
        check_content_depth(pages),
        check_faq_structure(pages),
        check_internal_link(pages),
        check_knowledge_graph(pages),
    ]
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.grade))


def to_context(findings: list[Finding]) -> str:
    """渲染成供 Agent 阅读的文本。"""
    lines = []
    for f in findings:
        if f.severity == "ok":
            continue
        lines.append(f"[{f.grade}档/{f.severity}] {f.title}：{f.fact}")
        if f.detail:
            lines.append(f"    {f.detail}")
    return "\n".join(lines)


def main() -> None:
    pages = load()
    findings = scan(pages)

    SCAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCAN_JSON.write_text(
        json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    icons = {"high": "🔴", "medium": "🟡", "low": "🔵", "ok": "🟢"}
    print(f"扫描 {len(pages)} 页，产出 {len(findings)} 条结构发现\n")
    for f in findings:
        print(f"{icons[f.severity]} [{f.grade}档] {f.title}")
        print(f"    {f.fact}")
        if f.detail:
            print(f"    {f.detail[:88]}")
        print()
    print(f"已写入 {SCAN_JSON}")


if __name__ == "__main__":
    main()
