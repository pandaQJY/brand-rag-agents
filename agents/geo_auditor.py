"""GEO 诊断 Agent —— 语义层判断，确定性指标由 scan.py 提供。"""

from __future__ import annotations

from agents.base import Agent, Blackboard, register
from agents.context import outline_budget, sample_bodies, site_outline
from agents.scan import scan, to_context
from agents.standard import apply_to


@register
class GeoAuditor(Agent):
    name = "geo_auditor"
    label = "GEO 诊断"
    description = "判断官网内容对 AI 的友好度：哪些已表达清楚、哪些会被忽略或误解、缺什么"
    depends_on = ("site_analyst",)
    depends_why = {"site_analyst": "诊断需要先有品牌定位与产品能力的基线，否则无法判断表达是否偏离"}
    prompt_file = "agents/geo_auditor"
    max_tokens = 2560

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        findings = apply_to(scan())
        bb.facts["scan"] = findings
        return {
            "brand_analysis": bb.text_of("site_analyst", "（上游未提供）"),
            # 本环节要回答「官网还缺什么」——那是对全站的断言，
            # 只给几篇正文样本就问「缺不缺」，得到的只能是猜测。
            "outline": site_outline(outline_budget()),
            "scan": to_context(findings),
            "samples": sample_bodies(4),
        }
