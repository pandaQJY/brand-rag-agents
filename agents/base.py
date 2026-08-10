"""Agent 基类与 Blackboard 共享状态。

设计要点：

1. **Agent 只声明「我依赖谁」，不直接调用彼此。**
   依赖是类属性，因此路由器可以在不执行任何 Agent 的前提下静态推导出完整
   执行计划——这是「调度」区别于「串 Prompt」的技术前提。

2. **中间结果统一写入 Blackboard。**
   多 Agent 系统最难调试的就是「中间发生了什么」，因此每个环节的产出都留在
   板上而非即用即抛；连续追问时已完成的 Agent 还可直接复用，无需重跑。

3. **输出采用行式格式而非 JSON。**
   写作链路实测：本项目所用模型在嵌套 JSON 上系统性失败（漏方括号、
   吐垃圾片段），改用行式格式后零失败。此处沿用同一结论。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import ClassVar

from core.llm import chat, load_prompt


# ── Blackboard ──────────────────────────────────────────────

@dataclass
class Blackboard:
    """Agent 之间的共享状态。

    键为 Agent 名，值为该 Agent 的产出。同时保留原始文本便于展示与调试。
    """

    question: str = ""
    results: dict[str, "AgentResult"] = field(default_factory=dict)
    facts: dict[str, object] = field(default_factory=dict)  # 非 Agent 产生的事实，如结构扫描
    # 本轮执行计划涵盖的 Agent。板上累积的是历轮结果，而「这一轮该拿哪些作答」
    # 由计划决定，两者不是一回事——见 in_scope()。
    scope: list[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return name in self.results

    def get(self, name: str) -> "AgentResult | None":
        return self.results.get(name)

    def text_of(self, name: str, default: str = "") -> str:
        r = self.results.get(name)
        return r.text if r else default

    def put(self, result: "AgentResult") -> None:
        self.results[result.agent] = result

    def reset_from(self, name: str) -> None:
        """使某个 Agent 及其下游失效（追问需要重算时用）。"""
        self.results.pop(name, None)

    def in_scope(self, exclude: str = "") -> dict[str, "AgentResult"]:
        """本轮执行计划涵盖的结果。

        追问时板上留着历轮产出，但取材范围必须限定在本轮计划内。
        否则会出现：计划只含官网信息分析，最终答案却引用了上一轮的 GEO 诊断——
        界面显示「1 个上游 Agent」，答案里却标着另一个环节的出处，自相矛盾。

        scope 未设置时退化为全部，便于单独实例化 Agent 调试。
        """
        names = set(self.scope) if self.scope else set(self.results)
        return {k: v for k, v in self.results.items() if k in names and k != exclude}


@dataclass
class AgentResult:
    agent: str
    label: str
    text: str  # Agent 的原始输出
    sections: dict[str, list[str]] = field(default_factory=dict)  # 解析后的结构
    elapsed: float = 0.0
    model: str = ""

    def section(self, name: str) -> list[str]:
        return self.sections.get(name, [])


# ── 行式输出解析 ─────────────────────────────────────────────

def parse_sections(text: str) -> dict[str, list[str]]:
    """解析 Agent 的行式输出。

    格式约定：`## 小节名` 起一节，其下每行一条内容。
    不用 JSON 的理由见模块 docstring。
    """
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("#"):
            current = line.lstrip("# ").strip()
            out.setdefault(current, [])
            continue
        if current is None:
            continue
        item = line.lstrip("-*·").strip()
        if item:
            out[current].append(item)
    return {k: v for k, v in out.items() if v}


# ── Agent 基类 ───────────────────────────────────────────────

class Agent:
    """所有 Agent 的基类。

    子类需声明：name / label / description / depends_on / prompt_file，
    并实现 build_context()——把 Blackboard 中的上游结果组装成 Prompt 变量。
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    description: ClassVar[str] = ""  # 供 LLM Router 判断用
    depends_on: ClassVar[tuple[str, ...]] = ()
    # 每个依赖「为什么必要」。执行计划面板直接展示这句话——
    # 只说「是 X 的前置依赖」是在陈述图的边，看的人仍然会问「那又怎样」。
    # 依赖的正当性属于业务判断，应当由声明它的 Agent 自己讲清楚。
    depends_why: ClassVar[dict[str, str]] = {}
    prompt_file: ClassVar[str] = ""
    task: ClassVar[str] = "agent"  # llm.py 的按任务选型键
    max_tokens: ClassVar[int] = 2048

    # 产出是否随「用户问了什么」而变。
    #
    # 连续追问时编排器会复用板上已有的结果，这个优化成立的前提是
    # **Agent 的产出与问题无关**——四个分析型 Agent 都只通读官网、
    # 输出固定小节，换个问题结论也一样，因此可以放心复用。
    #
    # 但原文问答与整合收口不是这样：它们的输入里就含用户的问题，
    # 复用上一轮的产出等于答非所问。这类 Agent 声明 True，每轮强制重跑。
    question_dependent: ClassVar[bool] = False

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        """返回填充 Prompt 所需的变量。子类实现。"""
        raise NotImplementedError

    def run(self, bb: Blackboard) -> AgentResult:
        t0 = time.perf_counter()
        prompt = load_prompt(self.prompt_file, **self.build_context(bb))
        text = chat(
            [{"role": "user", "content": prompt}],
            task=self.task,
            max_tokens=self.max_tokens,
        ).strip()
        result = AgentResult(
            agent=self.name,
            label=self.label,
            text=text,
            sections=parse_sections(text),
            elapsed=time.perf_counter() - t0,
        )
        bb.put(result)
        return result


# ── 注册表 ───────────────────────────────────────────────────

REGISTRY: dict[str, Agent] = {}


def register(agent_cls: type[Agent]) -> type[Agent]:
    """装饰器：把 Agent 登记到全局注册表，供路由与依赖解析使用。"""
    REGISTRY[agent_cls.name] = agent_cls()
    return agent_cls


def get(name: str) -> Agent:
    if name not in REGISTRY:
        raise KeyError(f"未注册的 Agent：{name}，可选 {list(REGISTRY)}")
    return REGISTRY[name]


def all_agents() -> list[Agent]:
    return list(REGISTRY.values())
