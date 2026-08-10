"""内容策略 Agent —— 同时消费诊断结果与客户问题，产出带优先级的建议。"""

from __future__ import annotations

from agents.base import Agent, Blackboard, register


@register
class ContentStrategist(Agent):
    name = "content_strategist"
    label = "内容策略"
    description = "根据诊断与客户问题，给出应补充的 FAQ / Blog / 案例 / 产品说明及优先级"
    depends_on = ("geo_auditor", "query_generator")
    depends_why = {
        "geo_auditor": "先知道现状缺什么，才谈得上补什么",
        "query_generator": "诊断只说明官网缺什么，不说明客户想看什么。"
                           "GEO 的内容是为了在客户向 AI 提问时被引用而写的——"
                           "没有真实问题清单，建议会退化成凭空补目录",
    }
    prompt_file = "agents/content_strategist"
    max_tokens = 2560

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        return {
            "brand_analysis": bb.text_of("site_analyst", "（未执行）"),
            "audit": bb.text_of("geo_auditor", "（未执行）"),
            "queries": bb.text_of("query_generator", "（未执行）"),
        }
