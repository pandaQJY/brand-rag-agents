"""Streamlit Demo：RAG 品牌内容写作系统。

设计目标不是"能生成一篇文章"，而是**把流水线的每一步摊开给人看**——
尤其是四层幻觉拦截与引用溯源，这两处是本方案区别于朴素 RAG 的地方。

运行：
    streamlit run apps/app_writer.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# Streamlit 把 sys.path[0] 设为脚本所在目录（apps/），而非仓库根目录，
# 因此 core / kb / writer 这些顶层包不可见。CLI 走 `python -m apps.cli`
# 没有这个问题，但 streamlit 不支持 -m，只能在入口处补一行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from core.config import BRAND_NAME, DATA_DIR  # noqa: E402
from kb.chunk import load_chunks  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from writer.content_types import CONTENT_TYPES  # noqa: E402
from writer.generate import CITE_RE, generate  # noqa: E402
from writer.intent import parse_request  # noqa: E402
from writer.literal import check as literal_check  # noqa: E402
from writer.literal import summarize as literal_summarize  # noqa: E402
from writer.outline import plan  # noqa: E402
from writer.retrieve import build_evidence  # noqa: E402
from writer.verify import LABELS, verify  # noqa: E402

st.set_page_config(page_title="品牌知识库 RAG 写作", page_icon="📝", layout="wide")

VERDICT_STYLE = {
    "supported": ("有据支持", "#17795A", "#E1F2EA"),
    "miscited": ("引用指错页", "#0B6E86", "#DDF0F5"),
    "unsupported": ("无证据支撑", "#A26A05", "#FBF0DA"),
    "contradicted": ("与证据矛盾", "#BC2E2E", "#FAE6E4"),
}

st.markdown(
    """
<style>
  .cite-badge{background:#E7ECFC;color:#1F44D6;padding:1px 6px;border-radius:3px;
    font-family:ui-monospace,Menlo,monospace;font-size:.82em;font-weight:600;white-space:nowrap}
  /* 第 3 层判定为捏造的引用直接标红，让人一眼看到是哪一句出的问题 */
  .cite-bad{background:#FAE6E4;color:#BC2E2E;padding:1px 6px;border-radius:3px;
    font-family:ui-monospace,Menlo,monospace;font-size:.82em;font-weight:700;
    white-space:nowrap;text-decoration:line-through}
  /* 凡设了固定浅色背景的元素，必须同时显式指定深色文字——
     否则深色主题下文字继承为白色，白字压浅底完全不可读。 */
  .risk-line{border-left:3px solid #BC2E2E;background:#FAE6E4;padding:9px 13px;
    border-radius:0 4px 4px 0;margin:7px 0;color:#1A1F28}
  .risk-line .reason{font-size:.85em;color:#4A5666}
  .stat-card{padding:11px 8px;border-radius:6px;text-align:center;line-height:1.45}
  .stat-card .n{font-size:1.35em;font-weight:700}
  .stat-card .t{font-size:.8em}
  .ev-card{border:1px solid rgba(128,140,160,.35);border-radius:6px;
    padding:10px 12px;margin-bottom:8px}
  .ev-id{font-family:ui-monospace,Menlo,monospace;font-size:.8em;font-weight:600;
    color:#1F44D6;background:#E7ECFC;padding:1px 6px;border-radius:3px}
  .src-tag{font-family:ui-monospace,Menlo,monospace;font-size:.72em;color:#77828F}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="加载检索索引与向量模型…")
def get_retriever() -> HybridRetriever:
    return HybridRetriever.load()


@st.cache_resource
def get_chunk_map() -> dict:
    return {c.chunk_id: c for c in load_chunks()}


def badge_citations(md: str, invalid: set[str] | None = None) -> str:
    """把 [Pxx] 渲染成徽章；第 3 层判定为捏造的编号标红划线。"""
    invalid = invalid or set()

    def _one(m):
        cid = m.group(1)
        if cid in invalid:
            return f'<span class="cite-bad" title="该编号不在本次证据集中，属捏造引用">{cid}</span>'
        return f'<span class="cite-badge">{cid}</span>'

    return CITE_RE.sub(_one, md)


# ─────────────────────────── 侧边栏 ───────────────────────────
with st.sidebar:
    st.title("📝 RAG 内容写作")
    st.caption(f"基于{BRAND_NAME}品宣 PDF · 42 页 / 38 个检索单元")

    instruction = st.text_area(
        "写作指令",
        f'请基于{BRAND_NAME}品宣PDF，生成一篇官网Blog，主题是："为什么中国出海品牌需要进行GEO优化？"',
        height=110,
        help="直接粘贴一整句指令即可，系统会自动识别内容类型与主题。",
    )
    run = st.button("生成", type="primary", use_container_width=True)

    # 实时展示解析结果，允许人工纠正
    req = None
    if instruction.strip():
        try:
            req = parse_request(instruction, allow_model=False)  # 侧边栏只用规则，零延迟
        except ValueError:
            req = None
    if req:
        st.caption(
            f"识别为 **{req.content_type}**（{req.type_source}）· "
            f"主题：{req.topic[:30]}{'…' if len(req.topic) > 30 else ''}"
        )
        override = st.selectbox(
            "内容类型（识别有误时可修正）",
            CONTENT_TYPES,
            index=CONTENT_TYPES.index(req.content_type),
        )
    else:
        override = CONTENT_TYPES[0]

    st.divider()
    st.caption(
        "**流水线**\n\n"
        "1. 指令解析（规则优先，模型兜底）\n"
        "2. 大纲规划（含资料库目录）\n"
        "3. 分节混合检索 + RRF\n"
        "4. 带 `[Pxx]` 标记生成\n"
        "5. 四层幻觉拦截"
    )

# ─────────────────────────── 主流程 ───────────────────────────
if run:
    retriever = get_retriever()
    with st.status("运行流水线…", expanded=True) as status:
        st.write("① 解析写作指令")
        parsed = parse_request(instruction)  # 此处允许模型兜底
        content_type = override or parsed.content_type
        topic = parsed.topic
        st.write(f"　　内容类型：{content_type}　主题：{topic}")

        st.write("② 规划大纲，生成分节检索查询")
        outline = plan(topic, content_type)

        st.write(f"③ 分节检索（{len(outline.sections)} 个单元）")
        pack = build_evidence(outline, retriever)

        st.write(f"④ 生成正文（{len(pack.chunks)} 份证据）")
        article = generate(pack, content_type)

        st.write("⑤ 事实核查")
        report = verify(article, pack)
        status.update(label="完成", state="complete", expanded=False)

    st.session_state.update(outline=outline, pack=pack, article=article, report=report)

if "article" not in st.session_state:
    st.info("在左侧输入主题并点击「生成」。首次运行需加载向量模型，约十几秒。")
    st.stop()

outline = st.session_state["outline"]
pack = st.session_state["pack"]
article = st.session_state["article"]
report = st.session_state["report"]

# ─────────────────────────── 顶部指标 ───────────────────────────
st.subheader(outline.title)
c1, c2, c3, c4 = st.columns(4)
st_ = article.stats
c1.metric("正文字数", st_["字数"])
c2.metric("引用处数", st_["引用总数"], f"覆盖 {st_['引用页数']} 页")
c3.metric(
    "引用编号有效性",
    "通过" if not article.invalid_cites else f"{len(article.invalid_cites)} 处捏造",
    delta=None if not article.invalid_cites else "需处理",
    delta_color="inverse" if article.invalid_cites else "normal",
)
c4.metric(
    "事实核查通过率",
    f"{report.pass_rate:.0%}",
    f"{len(report.risky)} 条待处理" if report.risky else "全部有据",
    delta_color="inverse" if report.risky else "normal",
)

tab_art, tab_check, tab_ev, tab_plan = st.tabs(
    ["📄 文章", "🔍 事实核查", "📚 检索证据", "🗂 大纲与查询"]
)

# ─────────────────────────── 文章 ───────────────────────────
with tab_art:
    left, right = st.columns([3, 2])
    with left:
        if article.invalid_cites:
            st.error(
                f"第 3 层检出捏造引用 {article.invalid_cites}——正文中已用红色删除线标出。"
                "这些编号不在本次检索到的证据集内，属模型虚构。"
            )
        st.markdown(
            badge_citations(article.markdown, set(article.invalid_cites)),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("##### 引用溯源")
        st.caption("选择一个引用编号，查看它对应的 PDF 原页与解析内容。")
        cited = article.cited_ids or [c.chunk_id for c in pack.chunks]
        pick = st.selectbox("引用编号", cited, label_visibility="collapsed")
        chunk = get_chunk_map().get(pick)
        if chunk:
            st.caption(f"《{chunk.title}》 · {chunk.section or '前言'} · 第 {chunk.page_nos} 页")
            for p in chunk.image_paths:
                img = DATA_DIR / p
                if img.exists():
                    st.image(str(img), use_container_width=True)
            with st.expander("文字层（PDF 原始字符）", expanded=False):
                st.text(chunk.text_layer or "（无）")
            with st.expander("视觉转写（模型判读，需核查）", expanded=False):
                st.text(chunk.vision_text or "（无）")

# ─────────────────────────── 核查 ───────────────────────────
with tab_check:
    st.markdown("##### 四层拦截结果")
    l1, l2 = st.columns(2)
    l1.success("第 1 层 · Prompt 硬约束｜第 2 层 · 引用解析（纯代码）")
    if article.invalid_cites:
        l2.error(f"第 3 层 · 引用编号校验：捏造 {article.invalid_cites}")
    else:
        l2.success("第 3 层 · 引用编号校验：通过（纯代码，零成本）")

    nums = literal_check(article.markdown)
    ns = literal_summarize(nums)
    msg = (f"第 3.5 层 · 数字确定性校验：{len(nums)} 个量化表述 → "
           f"逐字命中 {ns['exact']}、取整 {ns['approx']}、编造 {ns['missing']}")
    if ns["missing"]:
        st.error(msg)
        for c in nums:
            if c.status == "missing":
                st.markdown(
                    f"<div class='risk-line'><b style='color:#BC2E2E'>编造数字</b> "
                    f"「{html.escape(c.value)}」<br>{html.escape(c.sentence[:80])}<br>"
                    f"<span class='reason'>{html.escape(c.note)}</span></div>",
                    unsafe_allow_html=True)
    elif ns["approx"]:
        st.warning(msg + "　（取整属可接受改写，建议人工确认）")
    else:
        st.success(msg + "　全部逐字命中语料")

    counts = report.counts
    cols = st.columns(4)
    for col, v in zip(cols, VERDICT_STYLE):
        name, fg, bg = VERDICT_STYLE[v]
        col.markdown(
            f"<div class='stat-card' style='background:{bg};color:{fg}'>"
            f"<div class='n'>{counts[v]}</div><div class='t'>{name}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### 逐条判定")
    if report.risky:
        st.markdown("**需要处理：**")
        for c in report.risky:
            name, fg, bg = VERDICT_STYLE[c.verdict]
            cited = " ".join(c.cited) if c.cited else "未标引用"
            st.markdown(
                f"<div class='risk-line' style='border-left-color:{fg};background:{bg}'>"
                f"<b style='color:{fg}'>{name}</b> · 引用：{html.escape(cited)}<br>"
                f"「{html.escape(c.text)}」<br>"
                f"<span class='reason'>判据：{html.escape(c.reason)}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("未发现无支撑或矛盾的陈述。")

    with st.expander(f"全部 {len(report.claims)} 条声明", expanded=False):
        st.dataframe(
            [
                {
                    "事实性陈述": c.text,
                    "引用": " ".join(c.cited) or "—",
                    "判定": VERDICT_STYLE[c.verdict][0],
                    "判据": c.reason,
                }
                for c in report.claims
            ],
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────── 证据 ───────────────────────────
with tab_ev:
    st.caption(
        f"共 {len(pack.chunks)} 份去重证据；未被引用的 {len(article.unused_evidence)} 份"
        f"（{' '.join(article.unused_evidence) or '无'}）"
    )
    for i, se in enumerate(pack.sections, 1):
        st.markdown(f"**{i}. {se.section.heading}**")
        for h in se.hits:
            used = "✅ 已引用" if h.chunk.chunk_id in article.cited_ids else "· 未引用"
            st.markdown(
                f"<div class='ev-card'><span class='ev-id'>{h.chunk.chunk_id}</span> "
                f"{html.escape(h.chunk.title[:44])} "
                f"<span class='src-tag'>[{h.sources}] {used}</span></div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────── 大纲 ───────────────────────────
with tab_plan:
    st.caption("大纲不只是文章骨架，它同时是**检索计划**——每节自带贴近资料措辞的查询。")
    st.markdown(f"**导语应覆盖**：{outline.lead or '—'}")
    for i, s in enumerate(outline.sections, 1):
        st.markdown(f"**{i}. {s.heading}**")
        if s.intent:
            st.caption(f"意图：{s.intent}")
        for q in s.queries:
            st.code(q, language=None)
    if outline.faq:
        st.markdown("**FAQ 问题**")
        for q in outline.faq:
            st.markdown(f"- {q}")
