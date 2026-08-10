"""官网信息分析 Agent —— DAG 的根节点，无前置依赖。"""

from __future__ import annotations

from agents.base import Agent, Blackboard, register
from agents.context import site_digest


@register
class SiteAnalyst(Agent):
    name = "site_analyst"
    label = "官网信息分析"
    description = (
        "通读官网，提取品牌定位、产品能力、技术关键词、服务对象与核心表达。"
        "凡是问「官网上关于某件事是怎么说的」——概念、术语、区别、主张——都归它"
    )
    depends_on = ()
    prompt_file = "agents/site_analyst"
    max_tokens = 2048

    def build_context(self, bb: Blackboard) -> dict[str, str]:
        return {"site": site_digest()}
