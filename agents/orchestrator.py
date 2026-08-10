"""编排器：按执行计划依次运行 Agent，结果写入 Blackboard。

职责很薄——真正的调度决策在 router.py 已经完成。这里只负责：
按序执行、记录耗时、失败降级、把中间结果收拢成可展示的结构。

失败降级是有意设计：链路中某个 Agent 失败时，写入一条占位结果而非抛出异常，
让下游仍能拿到「该环节数据不足」的信号继续跑完。对演示而言，
跑完并说明哪一步失败，比中途崩溃有用得多。

用法：
    python -m agents.orchestrator "官网该补充什么内容？"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from agents import base, registry  # noqa: F401  registry 触发注册
from agents.base import AgentResult, Blackboard
from agents.router import Plan, PlanStep, route
from agents.synthesize import Synthesizer

SYNTHESIZER = Synthesizer()


@dataclass
class RunReport:
    plan: Plan
    blackboard: Blackboard
    results: list[AgentResult] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def steps(self) -> list[AgentResult]:
        """中间结果：各能力 Agent 的产出。"""
        return [r for r in self.results if r.agent != SYNTHESIZER.name]

    @property
    def final(self) -> AgentResult | None:
        """最终结果。

        多 Agent 时是整合环节的产出；单 Agent 时无需整合，它自己就是答案。
        """
        return self.results[-1] if self.results else None


def _run_step(step: PlanStep, agent, bb: Blackboard, report: RunReport, on_step) -> None:
    # 连续追问时复用已有结果——但仅限产出与问题无关的 Agent，
    # 见 base.Agent.question_dependent。
    if bb.has(step.agent) and not agent.question_dependent:
        report.results.append(bb.get(step.agent))
        if on_step:
            on_step(step, bb.get(step.agent), True)
        return

    try:
        result = agent.run(bb)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        report.failed.append((step.agent, msg))
        result = AgentResult(
            agent=step.agent,
            label=step.label,
            text=f"（本环节执行失败，下游据此降级处理）\n原因：{msg}",
        )
        bb.put(result)
    report.results.append(result)
    if on_step:
        on_step(step, result, False)


def execute(plan: Plan, bb: Blackboard | None = None, on_step=None) -> RunReport:
    bb = bb or Blackboard(question=plan.question)
    # 每轮都要更新，不能写成 `bb.question or plan.question`——那样第一轮设过之后
    # 就永不更新，追问时收口环节会对齐到上一轮的问题。
    bb.question = plan.question
    # 取材范围限定在本轮计划内。板上留着历轮产出，但它们不属于这一轮的答案。
    bb.scope = plan.agent_names
    report = RunReport(plan=plan, blackboard=bb)
    t0 = time.perf_counter()

    for step in plan.steps:
        _run_step(step, base.get(step.agent), bb, report, on_step)

    # 固定收口：把能力形状的产出对齐到问题形状。详见 synthesize.py。
    #
    # 单 Agent 时也要跑。曾按「只有一份材料，整合无意义」跳过，那是旧设计的残留——
    # 当时本环节做的是重写，一份材料确实无需重写。现在它做的是**对齐**：
    # 官网信息分析固定输出 7 个小节，用户问「GEO 和 SEO 的区别」时只需要其中一两节，
    # 跳过就等于把 7 节整份倒给用户。**Agent 只有一个，不代表小节只有一节。**
    #
    # 追问时也必须重跑：上游结果可以复用，但「回答的是哪个问题」每轮都变了。
    if plan.steps:
        bb.reset_from(SYNTHESIZER.name)
        _run_step(
            PlanStep(
                agent=SYNTHESIZER.name,
                label=SYNTHESIZER.label,
                reason="编排器固定收口：把各环节产出对齐到用户实际提出的问题",
                origin="orchestrator",
            ),
            SYNTHESIZER,
            bb,
            report,
            on_step,
        )

    report.elapsed = time.perf_counter() - t0
    return report


def run(question: str, bb: Blackboard | None = None, allow_llm: bool = True) -> RunReport:
    return execute(route(question, allow_llm=allow_llm), bb)


def main() -> None:
    ap = argparse.ArgumentParser(description="端到端执行")
    ap.add_argument("question")
    ap.add_argument("--no-llm-route", action="store_true")
    args = ap.parse_args()

    plan = route(args.question, allow_llm=not args.no_llm_route)
    print(plan.describe())
    print("\n" + "═" * 72)

    def show(step, result, cached):
        tag = "（复用缓存）" if cached else f"{result.elapsed:.1f}s"
        print(f"\n▶ {result.label} {tag}")
        print("─" * 72)
        print(result.text[:1400])

    report = execute(plan, on_step=show)

    print("\n" + "═" * 72)
    print(f"完成：{len(report.results)} 个 Agent，耗时 {report.elapsed:.1f}s")
    if report.failed:
        print(f"失败 {len(report.failed)} 个：{report.failed}")


if __name__ == "__main__":
    main()
