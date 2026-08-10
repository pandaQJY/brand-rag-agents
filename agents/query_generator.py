"""用户问题生成 Agent —— 模拟目标客户在 AI 平台上的提问。"""

from __future__ import annotations

from agents.base import Agent, Blackboard, register
from agents.context import outline_budget, site_outline


@register
class QueryGenerator(Agent):
    name = "query_generator"
    label = "用户问题生成"
    description = "模拟目标客户在 ChatGPT / Perplexity / DeepSeek 等平台可能提出的问题"
    depends_on = ("site_analyst",)
    depends_why = {"site_analyst": "客户会问什么，取决于这家公司卖什么、卖给谁——先有画像才谈得上模拟提问"}
    prompt_file = "agents/query_generator"
    max_tokens = 2048

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        return {
            "brand_analysis": bb.text_of("site_analyst", "（上游未提供）"),
            # 「官网未作答的问题」同样是对全站下断言，需要全站目录才能核对
            "outline": site_outline(outline_budget()),
        }
