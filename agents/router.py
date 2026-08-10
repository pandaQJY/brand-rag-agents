"""路由与调度：把一句话变成有序的执行计划。

这是「简单写多个 Prompt」与「真正调度」的分界。
核心主张：**路由的产物不是「选中一个 Agent」，而是一份执行计划**。

三层结构，沿用写作链路已验证的「确定性优先」思路：

    ① 规则匹配     关键词 → Agent          零成本、零延迟
    ② LLM Router   复合与模糊问题才触发     一次小调用
    ③ 依赖补齐     递归展开 + 拓扑排序      纯代码，确定性

第 ③ 层是关键：用户只说「该补充什么内容」，规则命中 ContentStrategist，
而它依赖 GeoAuditor 与 QueryGenerator，二者又都依赖 SiteAnalyst——
系统自行推导出四个 Agent 的执行顺序。**用户提一个需求，系统排一条流水线。**

每个 Agent 都附带被选中的理由，Demo 中直接展示：让人看见系统怎么想的，
而不只是它做了什么。

用法：
    python -m agents.router "官网哪些内容适合被 AI 引用？"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from agents import base
from core.llm import chat

# Agent → 触发词组。每个元组是一个 AND 组：组内全部出现才算命中，
# 任一组命中即选中该 Agent。
#
# 为什么不用子串匹配：初版写「适合被引用」，而真实提问是
# 「官网哪些内容适合被 AI 引用」——中间隔了「AI」，子串直接失效。
# 词组共现对措辞变形的抗性高得多，且仍是零成本的纯代码判断。
RULES: list[tuple[str, list[tuple[str, ...]]]] = [
    ("content_strategist", [
        ("补充", "内容"), ("该写",), ("写什么",), ("内容策略",), ("优化方向",),
        ("优先级",), ("该补",), ("内容规划",), ("建议", "内容"),
    ]),
    ("query_generator", [
        ("会问",), ("怎么问",), ("提问",), ("问题清单",), ("目标客户", "问"),
        ("会搜",), ("问哪些",), ("模拟", "问"), ("用户", "提问"),
    ]),
    ("geo_auditor", [
        # 曾写作 ("哪些","问题")，被「目标客户会问哪些问题」误命中——
        # 那问的是客户的提问，不是官网的毛病。收紧成连续词组即可区分。
        ("诊断",), ("体检",), ("哪里", "优化"), ("有哪些问题",),
        ("适合", "引用"), ("能否", "引用"), ("被引用",), ("AI友好",),
        ("忽略",), ("误解",), ("缺什么",), ("缺失",), ("优化建议",),
        ("哪些", "改"), ("问题", "在哪"),
    ]),
    # 「问内容」类：用户要的是官网怎么讲某件具体的事。
    #
    # 这几组词原本指向 site_analyst，因为当时没有别的地方可去——而
    # site_analyst 的八个小节里并没有「官网对某概念是怎么讲的」这一节，
    # 于是「GEO 和 SEO 有什么区别」得到的是一份品牌分析报告，答非所问。
    # 有了 site_qa（检索原文后作答）之后，这类问题才有了正确的落点。
    ("site_qa", [
        ("区别",), ("什么是",), ("怎么讲",), ("怎么说",), ("如何定义",),
        ("有没有讲",), ("提到",), ("官网", "解释"), ("原话",),
        ("怎么描述",), ("是什么意思",),
    ]),
    ("site_analyst", [
        ("表达", "什么"), ("什么公司",), ("介绍",), ("官网", "内容"),
        ("品牌定位",), ("产品能力",), ("服务对象",), ("摘要",),
        ("分析", "官网"), ("是做什么",), ("讲了什么",),
    ]),
]


_ROUTER_PROMPT = """判断下面这个用户问题需要调用哪些分析能力，只输出 Agent 名，每行一个。

可选 Agent：
{catalog}

规则：
- 只输出确实需要的，不要为了周全全选。
- 前置依赖不用你操心，系统会自动补齐——只输出用户直接想要的那个（或那几个）。
- 若问题同时涉及多个方面，可输出多个。
- **分清三类问法**，这是最容易搞混的地方：
  - 问**官网上某件具体的事是怎么说的**——「X 是什么」「X 和 Y 有什么区别」
    「官网有没有讲过 Z」——归 `site_qa`。它会检索原文后据实作答并标出处。
  - 问**这家公司整体是什么样**——「是家什么公司」「品牌定位」「服务对象」
    「官网整体表达了什么」——归 `site_analyst`。要的是通读全站得出的基线。
  - 问**官网做得怎么样**——「哪里有问题」「能不能被 AI 引用」「该改什么」
    ——才归 `geo_auditor`。
  只因为问题里出现了 GEO、SEO、AI 这类词就选诊断，是典型误判。
  同理，具体问题不要甩给 `site_analyst`：它输出的是固定小节的分析报告，
  回答不了「这个词官网怎么定义」。
- 除 Agent 名外不要输出任何其他文字。

用户问题：{question}"""


@dataclass
class PlanStep:
    agent: str
    label: str
    reason: str
    origin: str  # rule | llm | dependency


@dataclass
class Plan:
    question: str
    steps: list[PlanStep] = field(default_factory=list)
    route_source: str = "rule"  # rule | llm | fallback

    @property
    def agent_names(self) -> list[str]:
        return [s.agent for s in self.steps]

    def describe(self) -> str:
        lines = [f"问题：{self.question}", f"路由方式：{self.route_source}", "执行计划："]
        for i, s in enumerate(self.steps, 1):
            tag = {"rule": "规则命中", "llm": "模型判断", "dependency": "依赖补齐"}[s.origin]
            lines.append(f"  {i}. {s.label}（{s.agent}）")
            lines.append(f"     [{tag}] {s.reason}")
        return "\n".join(lines)


# ── ① 规则匹配 ───────────────────────────────────────────────

def match_rules(question: str) -> list[tuple[str, str]]:
    """词组共现匹配。返回 [(agent_name, 命中说明)]。"""
    q = question.lower()
    hits = []
    for agent, groups in RULES:
        for group in groups:
            if all(term.lower() in q for term in group):
                desc = "「" + "」+「".join(group) + "」" if len(group) > 1 else f"「{group[0]}」"
                hits.append((agent, desc))
                break
    return hits


# ── ② LLM 兜底 ───────────────────────────────────────────────

def ask_llm(question: str) -> list[str]:
    catalog = "\n".join(
        f"- {a.name}（{a.label}）：{a.description}" for a in base.all_agents()
    )
    raw = chat(
        [{"role": "user", "content": _ROUTER_PROMPT.format(catalog=catalog, question=question)}],
        task="route",
        max_tokens=200,
    )
    valid = {a.name for a in base.all_agents()}
    out = []
    for line in raw.splitlines():
        name = line.strip().lstrip("-*·0123456789. ").split("（")[0].split("(")[0].strip()
        if name in valid and name not in out:
            out.append(name)
    return out


# ── ③ 依赖补齐与拓扑排序 ─────────────────────────────────────

def expand_dependencies(targets: list[str]) -> dict[str, str]:
    """递归补齐前置 Agent。

    返回 {agent_name: 补齐来源}——直接命中的记为空串，被补齐的记为下游 Agent 名。
    这是「调度」的核心动作，且完全由代码完成：依赖是声明在 Agent 类上的静态属性，
    递归展开无需模型参与，因而零成本、零波动、结果可复现。
    """
    origin: dict[str, str] = {t: "" for t in targets}
    stack = list(targets)
    while stack:
        name = stack.pop()
        for dep in base.get(name).depends_on:
            if dep not in origin:
                origin[dep] = name
                stack.append(dep)
    return origin


def topological_order(names: set[str]) -> list[str]:
    """按依赖关系排序。同层保持注册顺序，使输出稳定可复现。"""
    ordered: list[str] = []
    remaining = set(names)
    guard = 0
    while remaining:
        guard += 1
        if guard > 100:
            raise RuntimeError(f"依赖图存在环：{remaining}")
        ready = [
            n for n in base.REGISTRY
            if n in remaining and all(d in ordered for d in base.get(n).depends_on)
        ]
        if not ready:
            raise RuntimeError(f"依赖无法满足：{remaining}")
        ordered.extend(ready)
        remaining -= set(ready)
    return ordered


# ── 主入口 ───────────────────────────────────────────────────

def route(question: str, allow_llm: bool = True) -> Plan:
    plan = Plan(question=question)

    hits = match_rules(question)
    reasons: dict[str, tuple[str, str]] = {}  # agent → (origin, reason)

    if hits:
        plan.route_source = "rule"
        for agent, kw in hits:
            reasons[agent] = ("rule", f"问题中共现词组 {kw}")
    elif allow_llm:
        plan.route_source = "llm"
        for agent in ask_llm(question):
            reasons[agent] = ("llm", "规则未命中，由模型判断该问题需要此能力")

    if not reasons:
        # 兜底：什么都没匹配上时，做一次完整分析总比拒答好
        plan.route_source = "fallback"
        reasons["geo_auditor"] = ("llm", "未能识别意图，默认执行完整诊断")

    expanded = expand_dependencies(list(reasons))
    for name, by in expanded.items():
        if name not in reasons:
            downstream = base.get(by)
            why = downstream.depends_why.get(name, "")
            reasons[name] = (
                "dependency",
                f"{downstream.label} 需要它：{why}" if why
                else f"作为 {downstream.label} 的前置依赖被自动补齐",
            )

    for name in topological_order(set(expanded)):
        origin, reason = reasons[name]
        plan.steps.append(
            PlanStep(agent=name, label=base.get(name).label, reason=reason, origin=origin)
        )
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="路由：把问题变成执行计划")
    ap.add_argument("question")
    ap.add_argument("--no-llm", action="store_true", help="只用规则，不调模型")
    args = ap.parse_args()

    import agents.registry  # noqa: F401  触发 Agent 注册

    print(route(args.question, allow_llm=not args.no_llm).describe())


if __name__ == "__main__":
    main()
