"""导入全部 Agent 以触发注册。路由与编排只需 import 本模块。

导入顺序即注册顺序。依赖关系由 router.topological_order 保证，
注册顺序只用于**同层节点之间**的先后——因此顺序本身不影响正确性，
但必须固定，否则执行计划的展示顺序会在不同入口下漂移。
按字母序排列即可满足这一点。
"""

from agents import (  # noqa: F401
    content_strategist,
    geo_auditor,
    query_generator,
    site_analyst,
    site_qa,
)
