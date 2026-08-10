"""导入全部 Agent 以触发注册。路由与编排只需 import 本模块。

导入顺序即注册顺序，而注册顺序决定拓扑排序中同层节点的先后
（见 router.topological_order），因此这里保持「先基线、后派生」的排列，
使执行计划的展示顺序稳定可复现。
"""

from agents import (  # noqa: F401
    content_strategist,
    geo_auditor,
    query_generator,
    site_analyst,
    site_qa,
)
