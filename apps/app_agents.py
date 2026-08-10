"""Streamlit Demo：GEO 智能体协作系统。

设计目标不是「能出结果」，而是**让人看见系统怎么决策**——
执行计划、每个 Agent 被选中的理由、依赖补齐的过程、各环节的中间结果，全部摊开。

连续追问的复用由两处配合而成，Blackboard 只是其中被持久化的那个容器：
    1. 本模块把同一个 Blackboard 存进 st.session_state，跨轮沿用；
    2. orchestrator._run_step 发现板上已有结果就跳过，不重复调用模型。

成立的前提是**分析型 Agent 的产出与用户问什么无关**（它们的 build_context
都不读 bb.question），所以隔轮复用不会串味。读它的环节——原文问答与整合收口
——声明 `question_dependent = True`，每轮强制重跑，侧栏的缓存清单据此把它们
排除在外。这条判据写在 Agent 类上而非硬编码在编排器里，
新增 Agent 时只需如实声明，调度与界面自动跟上。

运行：
    streamlit run apps/app_agents.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# Streamlit 把 sys.path[0] 设为脚本所在目录（apps/），而非仓库根目录，
# 因此 agents / core / retrieval 这些顶层包不可见。CLI 走 `python -m apps.cli`
# 没有这个问题，但 streamlit 不支持 -m，只能在入口处补一行。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from agents import base, registry  # noqa: F401  registry 触发 Agent 注册
from core.config import BRAND_NAME, site_domain  # noqa: E402
from agents.base import Blackboard  # noqa: E402
from agents.orchestrator import SYNTHESIZER, execute  # noqa: E402
from agents.router import route  # noqa: E402
from agents.scan import scan  # noqa: E402
from agents.standard import apply_to  # noqa: E402
from website.fetch import load  # noqa: E402

st.set_page_config(page_title="GEO 智能体协作系统", page_icon="🧭", layout="wide")

ORIGIN_STYLE = {
    "rule": ("规则命中", "#17795A", "#E1F2EA"),
    "llm": ("模型判断", "#1F44D6", "#E7ECFC"),
    "dependency": ("依赖补齐", "#A26A05", "#FBF0DA"),
}
GRADE_STYLE = {
    "A": ("A 档 · 违反自述标准", "#BC2E2E", "#FAE6E4"),
    "B": ("B 档 · 可观测事实", "#1F44D6", "#E7ECFC"),
    "C": ("C 档 · 行业推测", "#A26A05", "#FBF0DA"),
}
# 档位标记会出现在最终答案里。对第一次用的人，「[B档]」三个字不解释就是天书——
# 证据分级是本系统的核心设计，恰恰最不该让人猜。悬停给全文，行内给图例。
GRADE_HINT = {
    "A": f"判据来自{BRAND_NAME}自己的品宣稿：他们写明的标准，官网没做到。最有力",
    "B": "代码扫描出的可观测事实，结果可复现",
    "C": "行业推测，无公开确证，仅供参考",
}
_GRADE_RE = re.compile(r"\[([ABC])档\]")

st.markdown(
    """
<style>
  /* 凡设固定浅色背景处必须显式指定深色文字，否则深色主题下白字压浅底不可读 */
  .chip{display:inline-block;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;
    font-weight:600;padding:2px 8px;border-radius:99px;white-space:nowrap}
  .step{border:1px solid rgba(128,140,160,.35);border-radius:6px;
    padding:9px 12px;margin-bottom:7px}
  .step .name{font-weight:700;font-size:14px}
  .step .why{font-size:12.5px;opacity:.85;margin-top:3px}
  .finding{border-left:3px solid #1F44D6;background:#F2F5F9;color:#1A1F28;
    padding:8px 12px;border-radius:0 4px 4px 0;margin:6px 0;font-size:13px}
  .finding .src{font-size:11.5px;color:#4A5666;margin-top:4px}
  .arrow{color:#77828F;font-family:ui-monospace,Menlo,monospace;margin:0 5px}
  .grade{display:inline-block;font-size:10.5px;font-weight:700;line-height:1.5;
    padding:0 5px;border:1px solid;border-radius:4px;margin-right:5px;
    vertical-align:1px;cursor:help;white-space:nowrap}
  .legend{font-size:12px;line-height:1.85;margin:10px 0 14px;padding:9px 12px;
    border:1px dashed rgba(128,140,160,.5);border-radius:6px}
  /* 最终结果的分隔标题：不设背景，只用色条与字重，深浅色主题下都可读 */
  .final-head{border-left:4px solid #E8563F;padding:2px 0 2px 11px;margin:18px 0 2px;
    font-size:17px;font-weight:700;letter-spacing:.02em}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def site_stats() -> dict:
    pages = load()
    return {"pages": len(pages), "chars": sum(p.char_count for p in pages)}


@st.cache_data(show_spinner=False)
def scan_findings():
    return apply_to(scan())


def chip(text: str, fg: str, bg: str) -> str:
    return f'<span class="chip" style="color:{fg};background:{bg}">{text}</span>'


def render_plan(plan) -> None:
    """执行计划——本 Demo 最重要的展示区。"""
    st.markdown(
        f"**路由方式**：{ {'rule': '规则匹配（零成本）', 'llm': 'LLM Router 兜底', 'fallback': '未识别意图，默认完整诊断'}[plan.route_source] }"
    )
    direct = sum(1 for s in plan.steps if s.origin != "dependency")
    auto = len(plan.steps) - direct
    if auto:
        st.caption(
            f"用户需求直接命中 {direct} 个 Agent，系统自动补齐 {auto} 个前置依赖，"
            f"并按拓扑序排定执行顺序。"
        )
    for i, s in enumerate(plan.steps, 1):
        label, fg, bg = ORIGIN_STYLE[s.origin]
        st.markdown(
            f'<div class="step"><span class="name">{i}. {html.escape(s.label)}</span> '
            f'{chip(label, fg, bg)}<div class="why">{html.escape(s.reason)}</div></div>',
            unsafe_allow_html=True,
        )


def with_grade_chips(text: str) -> str:
    """把 `[B档]` 换成带悬停说明的彩色小标。"""
    def sub(m):
        g = m.group(1)
        _, fg, _ = GRADE_STYLE[g]
        return (f'<span class="grade" style="color:{fg};border-color:{fg}" '
                f'title="{GRADE_HINT[g]}">{g} 档</span>')
    return _GRADE_RE.sub(sub, text)


def grade_legend(text: str) -> None:
    """只在正文真的出现档位标记时才给图例——没出现就别添无关信息。"""
    used = sorted(set(_GRADE_RE.findall(text)))
    if not used:
        return
    parts = []
    for g in used:
        _, fg, _ = GRADE_STYLE[g]
        parts.append(
            f'<span class="grade" style="color:{fg};border-color:{fg}">{g} 档</span>'
            f'<span style="opacity:.75">{html.escape(GRADE_HINT[g])}</span>'
        )
    st.markdown(
        '<div class="legend"><b>证据分级</b>（结论的可信度依据，'
        '来自「GEO 诊断」环节的确定性扫描）<br>' + "<br>".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def render_scan() -> None:
    """确定性扫描结果——GEO 诊断环节的输入，也是答案里 A/B/C 档位的出处。"""
    risky = [f for f in scan_findings() if f.severity != "ok"]
    st.caption(
        f"纯代码扫描全站，{len(risky)} 项有问题。零模型调用，结果可复现——"
        "答案里的 [A档]/[B档]/[C档] 都追溯到这里。"
        f"**A 档最硬**：判据是{BRAND_NAME}自己品宣稿写的标准，官网没做到。"
    )
    for f in risky:
        _, fg, _ = GRADE_STYLE[f.grade]
        st.markdown(
            f'<div class="finding" style="border-left-color:{fg}">'
            f'<b>[{f.grade}档] {html.escape(f.title)}</b><br>{html.escape(f.fact)}'
            f'<div class="src">{html.escape(f.detail[:200])}</div></div>',
            unsafe_allow_html=True,
        )


def render_results(results, cached: set[str]) -> None:
    """分两层展示：链路末端的产出是最终结果，其余是中间结果。

    初版把各 Agent 平铺成同级标签页，看的人无法判断哪个是最终答案——
    最后一个环节的产出才是整合结果，前面的都是它的输入。
    层级必须显式表达出来，否则「中间结果」与「结论」混作一团。
    """
    if not results:
        return
    *upstream, final = results

    if upstream:
        with st.expander(f"🔍 中间结果（{len(upstream)} 个上游 Agent）", expanded=False):
            st.caption("以下结果作为最终整合的输入，逐个可查。")
            tabs = st.tabs([r.label for r in upstream])
            for tab, r in zip(tabs, upstream):
                with tab:
                    st.caption(
                        "♻️ 复用缓存，未重复调用模型"
                        if r.agent in cached
                        else f"⏱ {r.elapsed:.1f}s"
                    )
                    # 确定性扫描只服务于诊断环节，故挂在它下面，而不是做成
                    # 全局面板——它与用户问什么无关，摆在侧边栏是喧宾夺主。
                    if r.agent == "geo_auditor":
                        out, inp = st.tabs(["语义判断（本环节输出）", "确定性扫描（本环节输入）"])
                        with out:
                            st.markdown(r.text)
                        with inp:
                            render_scan()
                    else:
                        st.markdown(r.text)

    st.markdown('<div class="final-head">最终结果</div>', unsafe_allow_html=True)
    st.caption(
        f"编排器收口环节：拆解你的问题，逐问从上述 {len(upstream)} 个环节中取证作答"
        if upstream
        else f"本次问题只需单个 Agent（{final.label}），无需整合"
    )
    grade_legend(final.text)
    st.markdown(with_grade_chips(final.text), unsafe_allow_html=True)


# ─────────────────────────── 侧边栏 ───────────────────────────
with st.sidebar:
    st.title("🧭 GEO 智能体协作")
    stats = site_stats()
    st.caption(f"分析对象：{site_domain()} · {stats['pages']} 页 / {stats['chars']:,} 字")

    st.divider()
    st.markdown("##### 会话")
    bb: Blackboard = st.session_state.setdefault("bb", Blackboard())
    # 产出随问题而变的环节不算可复用缓存——它们每轮必重跑
    # （见 base.Agent.question_dependent）。混进这份清单会让人以为最终答案
    # 也是复用的，而那恰好与它们的职责相反：要对齐的正是「你这轮问的是什么」。
    volatile = {SYNTHESIZER.name} | {a.name for a in base.all_agents() if a.question_dependent}
    cached = [r for name, r in bb.results.items() if name not in volatile]
    if cached:
        st.caption("已缓存的 Agent 结果（追问时复用，不重复调用）：")
        for r in cached:
            st.markdown(f"- {r.label} `{r.elapsed:.0f}s`")
        st.caption(
            f"追问命中可省 {sum(r.elapsed for r in cached):.0f}s。"
            f"每轮重跑的环节（{'、'.join(sorted(volatile))}）不在此列。"
        )
    else:
        st.caption("尚无缓存。首次提问会执行完整链路。")
    if st.button("清空会话", use_container_width=True):
        st.session_state.bb = Blackboard()
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(
        "**调度三层**\n\n"
        "1. 规则匹配（词组共现，零成本）\n"
        "2. LLM Router 兜底\n"
        "3. 依赖补齐 + 拓扑排序（纯代码）"
    )

# ─────────────────────────── 主区 ───────────────────────────
st.session_state.setdefault("history", [])

if not st.session_state.history:
    st.info(
        f"输入一个关于{BRAND_NAME}官网的问题。系统会先给出**执行计划**"
        "（调用哪些 Agent、为什么、什么顺序），再逐个执行并展示中间结果。\n\n"
        f"试试：`请分析{BRAND_NAME}官网目前哪些内容适合被 AI 引用，并给出 GEO 优化建议`"
    )

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        with st.expander("🗺 执行计划", expanded=False):
            render_plan(turn["plan"])
        render_results(turn["results"], turn["cached"])

question = st.chat_input("问点什么，比如：官网该补充什么内容？")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        bb = st.session_state.bb
        bb.question = question
        cached_before = set(bb.results)

        with st.spinner("路由中…"):
            plan = route(question)

        with st.expander("🗺 执行计划", expanded=True):
            render_plan(plan)

        progress = st.status(f"执行 {len(plan.steps)} 个 Agent…", expanded=True)

        # 复用与实际执行的条数在这里累加，而不是事后用 cached_before 去比对：
        # 上一轮的整合收口也在那个集合里，但它这轮被 reset_from 清掉重跑了，
        # 拿它统计会把「每轮必重跑的环节」误算成复用。on_step 收到的
        # was_cached 来自 _run_step 执行时的实际判断，是准的。
        tally = {"reused": 0, "ran": 0}

        def on_step(step, result, was_cached):
            tally["reused" if was_cached else "ran"] += 1
            with progress:
                if was_cached:
                    st.write(f"♻️ {result.label}　复用缓存")
                else:
                    st.write(f"✅ {result.label}　{result.elapsed:.1f}s")

        report = execute(plan, bb, on_step=on_step)
        # 面板完成后会折叠，这行标签是多数人唯一看到的信息。
        # 只报「N 个环节 + 总耗时」会读成「N 个环节跑了这么久」——
        # 而复用的那几个根本没调模型，两个数字必须拆开说。
        detail = (
            f"{tally['reused']} 个复用缓存、{tally['ran']} 个实际执行，"
            if tally["reused"]
            else ""
        )
        progress.update(
            label=f"完成：{len(report.results)} 个环节，{detail}共 {report.elapsed:.1f}s",
            state="complete",
            expanded=False,
        )

        if report.failed:
            st.error(f"以下 Agent 执行失败，下游已降级处理：{report.failed}")

        render_results(report.results, cached_before)

        st.session_state.history.append(
            {
                "question": question,
                "plan": plan,
                "results": report.results,
                "cached": cached_before,
            }
        )
