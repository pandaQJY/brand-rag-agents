"""整合环节 —— 把能力形状的产出，对齐回问题形状。

**它不参与路由，不出现在能力目录里，用户也点名调不到。**
这是有意的：前四个 Agent 是「能力」，由用户需求决定要不要跑；
本环节是编排器的固定收口动作（map-reduce 里的 reduce），
只要跑了 ≥2 个 Agent 就自动追加，与用户问什么无关。

## 为什么需要这一步

每个 Agent 的输出结构由它自己的职责决定——GeoAuditor 永远输出那四个小节，
ContentStrategist 永远输出三个优先级档。它们是**能力形状**的。
而用户的问题是**任意形状**的：一句话可能问两件事，措辞也各不相同。
把链路最后一个 Agent 的输出直接当答案，等于答非所问——
问「哪些内容适合被引用，并给出优化建议」，只答了后半问。

## 为什么由代码搬运，而不是让模型写

初版让模型读完全部材料自己写答案，连续踩了三个坑：
绕过专责环节自己重新推导建议、把优先级分档拍平、上游明明有 2 条只保留 1 条。
每次都靠往 Prompt 里加约束去堵，每次都堵不住——因为**「原样搬运」根本不是
生成任务**。模型每复述一遍就有一次改写和丢失的机会，而这种丢失是静默的：
用户看到的是一份残缺的清单，却不知道自己少看了什么。

因此改用写作链路已验证的分工——**凡能确定性裁决的，就不该交给模型**：

    模型负责  对齐：用户问的这件事，该由哪个环节的哪个小节回答
    代码负责  搬运：把那个小节整段原样取出

这样「漏条目」「改措辞」「丢档位标记」在结构上就不可能发生，
不需要任何 Prompt 约束去防。模型的输出只是一张对照表，
即使它出错，错的也只是「选错小节」——看得见、可核对，而不是悄悄少给两条。
"""

from __future__ import annotations

import re
import time

from agents.base import Agent, AgentResult, Blackboard
from core.llm import chat, load_prompt

_PICK = re.compile(r"^@采用\s+(\S+)\s*/\s*(.+?)\s*$")
_MISS = re.compile(r"^@缺失\s*(.*)$")
_HEAD = re.compile(r"^(#{1,6})\s*(.+?)\s*$")


def sections_of(text: str) -> list[str]:
    """列出一段 Agent 输出里的所有 `## 小节名`。"""
    return [m.group(2) for line in text.splitlines() if (m := _HEAD.match(line.strip()))]


def extract_section(text: str, name: str) -> str | None:
    """原样取出指定小节的正文（不含标题行）。

    命中规则：先精确匹配，再退化为去符号后的包含匹配——模型偶尔会把
    「已清晰表达」写成「已清晰表达的内容」，没必要为此让整条对齐失败。
    """
    lines = text.splitlines()
    norm = lambda s: re.sub(r"[\s　:：、,，。.]+", "", s)  # noqa: E731
    target = norm(name)

    start = level = None
    for i, line in enumerate(lines):
        m = _HEAD.match(line.strip())
        if not m:
            continue
        cur = norm(m.group(2))
        if cur == target or (start is None and (target in cur or cur in target)):
            start, level = i, len(m.group(1))
            if cur == target:
                break
    if start is None:
        return None

    body: list[str] = []
    for line in lines[start + 1:]:
        m = _HEAD.match(line.strip())
        if m and len(m.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body).strip() or None


class Synthesizer(Agent):
    name = "synthesize"
    label = "整合结果"
    description = "把各环节产出对齐到用户实际提出的问题（编排器固定收口，不参与路由）"
    depends_on = ()  # 不走依赖机制：材料是运行期才知道的，由编排器注入
    prompt_file = "agents/synthesize"
    task = "synthesize"
    max_tokens = 800  # 只产出一张对照表，不需要长输出
    question_dependent = True  # 每轮必须重跑：上游可复用，但「回答的是哪个问题」每轮都变

    def _sources(self, bb: Blackboard) -> dict[str, AgentResult]:
        """本轮计划涵盖的环节产出——不是板上的一切。见 Blackboard.in_scope()。"""
        return bb.in_scope(exclude=self.name)

    PREVIEW_CHARS = 260

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        """目录里必须带内容预览，不能只给小节名。

        初版只列小节名，追问「传统 SEO 与 GEO 的区别是什么」时判成「未覆盖」——
        可答案就在「可被AI引用的内容」那一节里。**小节名是按能力职责起的，
        不是按用户会问什么起的**，光看名字无从判断相关性。
        预览让它能按内容对齐，同时产出仍然只是指针，正文照样由代码原样搬运。
        """
        catalog = []
        for name, r in self._sources(bb).items():
            lines = [f"### {name}（{r.label}）"]
            heads = sections_of(r.text)
            for h in heads:
                body = re.sub(r"\s+", " ", extract_section(r.text, h) or "").strip()
                tail = "…" if len(body) > self.PREVIEW_CHARS else ""
                lines.append(f"- **{h}**：{body[:self.PREVIEW_CHARS]}{tail}")
            catalog.append("\n".join(lines) if heads else lines[0] + "\n- （无小节）")
        return {
            "question": bb.question or "（未提供原始问题）",
            "catalog": "\n\n".join(catalog) or "（无可用材料）",
        }

    def run(self, bb: Blackboard) -> AgentResult:
        t0 = time.perf_counter()
        plan_text = chat(
            [{"role": "user", "content": load_prompt(self.prompt_file, **self.build_context(bb))}],
            task=self.task,
            max_tokens=self.max_tokens,
        ).strip()

        result = AgentResult(
            agent=self.name,
            label=self.label,
            text=self._assemble(plan_text, self._sources(bb)),
            elapsed=time.perf_counter() - t0,
        )
        result.sections = {"对照表": plan_text.splitlines()}  # 留档，便于排查对齐错误
        bb.put(result)
        return result

    def _assemble(self, plan_text: str, sources: dict[str, AgentResult]) -> str:
        """按对照表拼装答案。正文一律原样搬运，本函数不生成任何内容。"""
        out: list[str] = []
        pending: list[str] = []  # 当前小节下待搬运的条目

        def flush():
            if pending:
                out.extend(pending)
                pending.clear()

        for raw in plan_text.splitlines():
            line = raw.strip()
            if not line or line.startswith("```"):
                continue

            if (m := _HEAD.match(line)) and not line.startswith("@"):
                flush()
                out.append(f"\n## {m.group(2)}\n")
                continue

            if m := _MISS.match(line):
                pending.append(f"> {m.group(1) or '现有分析未覆盖此问。'}\n")
                continue

            if m := _PICK.match(line):
                agent, sec = m.group(1), m.group(2)
                src = sources.get(agent)
                if src is None:  # 模型写了不存在的环节名
                    pending.append(f"> （对齐失败：没有名为 `{agent}` 的环节）\n")
                    continue
                body = src.text if sec == "*" else extract_section(src.text, sec)
                if not body:
                    pending.append(f"> （对齐失败：{src.label} 中没有「{sec}」小节）\n")
                    continue
                if sec != "*":
                    pending.append(f"**{sec}**\n")
                pending.append(f"{body}\n")
                pending.append(f"<sub>↑ 原样取自：{src.label}</sub>\n")
                continue

        flush()
        return "\n".join(out).strip() or "（对齐失败，未能生成整合结果）"
