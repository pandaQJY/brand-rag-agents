"""命令行入口。

    python -m apps.cli "官网该补充什么内容？"
    python -m apps.cli --interactive        # 连续追问模式
    python -m apps.cli --brief "……"         # 只要最终结果

**中间结果默认输出。** 初版把它藏在 --all 后面，结果是默认跑一遍只看得到
最终答案，看不见系统怎么想的——而这恰恰是多 Agent 系统最需要暴露的部分。
默认行为就该是完整行为，精简才需要显式索取。
"""

from __future__ import annotations

import argparse

from agents import registry  # noqa: F401
from agents.base import Blackboard
from agents.orchestrator import execute
from agents.router import route

TAG = {"rule": "规则命中", "llm": "模型判断", "dependency": "依赖补齐"}


def answer(question: str, bb: Blackboard, brief: bool = False) -> None:
    plan = route(question)
    print(f"\n路由方式：{plan.route_source}")
    print("执行计划：")
    for i, s in enumerate(plan.steps, 1):
        print(f"  {i}. {s.label}　[{TAG[s.origin]}] {s.reason}")

    cached = set(bb.results)
    print()

    def on_step(step, result, was_cached):
        mark = "♻️ 复用缓存" if was_cached else f"{result.elapsed:.1f}s"
        print(f"  ✓ {result.label}　{mark}")

    report = execute(plan, bb, on_step=on_step)

    if not report.results:
        return
    *upstream, final = report.results

    # 分两段打印，而不是把所有环节平铺成同级：链路最后一个环节的产出才是答案，
    # 前面几个是它的输入。平铺出来，读的人无从判断哪一份是结论。
    if upstream and not brief:
        print(f"\n{'═' * 70}")
        print(f"中间结果——{len(upstream)} 个上游环节，均为最终整合的输入")
        for r in upstream:
            tag = "（复用缓存）" if r.agent in cached else f"（{r.elapsed:.1f}s）"
            print(f"\n{'─' * 70}\n▶ {r.label}{tag}\n")
            print(r.text)

    print(f"\n{'═' * 70}\n最终结果\n{'═' * 70}")
    print(final.text)

    print(f"\n耗时 {report.elapsed:.1f}s")
    if report.failed:
        print(f"失败：{report.failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GEO 智能体协作系统")
    ap.add_argument("question", nargs="?", help="要问的问题")
    ap.add_argument("-i", "--interactive", action="store_true", help="连续追问模式")
    ap.add_argument("--brief", action="store_true", help="只输出最终结果，略去中间结果")
    args = ap.parse_args()

    bb = Blackboard()
    if args.interactive:
        print("连续追问模式，输入空行退出。已完成的 Agent 会被复用。")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            answer(q, bb, args.brief)
    elif args.question:
        answer(args.question, bb, args.brief)
    else:
        ap.error("需要提供问题，或使用 --interactive")


if __name__ == "__main__":
    main()
