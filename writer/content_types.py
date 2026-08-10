"""内容类型规格。

四种内容类型的**结构差异是实质性的**，不能只靠往 Prompt 里塞一个类型名来区分：
FAQ 是问答列表而非论述文，品牌介绍不需要导语和 FAQ，产品介绍应按能力模块组织。
把各自的结构规范集中在这里，分别注入大纲与生成两个 Prompt。

`Section` 这个数据结构对四种类型通用，只是语义不同：
    Blog / 品牌介绍 / 产品介绍 → 一个 Section 是一个板块
    FAQ                        → 一个 Section 是一个问题
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContentTypeSpec:
    name: str
    section_word: str  # 大纲里「小节」这一概念在该类型下的称呼
    outline_guide: str  # 注入 prompts/outline.md
    generate_guide: str  # 注入 prompts/generate.md


BLOG = ContentTypeSpec(
    name="官网Blog",
    section_word="小节",
    outline_guide="""这是一篇官网博客，读者带着「这事跟我有什么关系」的疑问进来。

- 规划 3–5 个小节，构成一条递进的论证线：现状变化 → 问题所在 → 解决路径 → 实证。
- 小节标题写成读者关心的问题或判断，不要写成资料的章节名。
- `lead` 写导语应当覆盖什么，用规划口吻。
- FAQ 可选，2–3 条，只列问题。""",
    generate_guide="""输出一篇散文体博客：

```
# 文章标题

导语段落（2–3 句，点出读者的实际处境，不写「本文将探讨」这类模板腔）。

## 小节标题

正文……[P12]

## 常见问题

**问题？**

答案……[P25]
```

- 每节 200–400 字，成段散文，不要罗列短句。
- 结尾 2–3 句收束，可自然引向了解产品，但不写成硬广。
- 若未提供 FAQ 问题，则省略「常见问题」整节。""",
)

FAQ = ContentTypeSpec(
    name="FAQ",
    section_word="问题",
    outline_guide="""这是一份 FAQ，**不是论述文**：没有导语、没有论证线，只有问答。

- 规划 6–10 个**问题**，每个 Section 的 `heading` 就是一个完整的问句。
- 问题要写成目标客户真会问出口的样子（「GEO 和 SEO 有什么区别？」），
  不要写成标题式短语（「GEO 与 SEO 对比」）。
- 覆盖面按「是什么 → 为什么 → 怎么做 → 多少钱 / 适合谁」铺开，避免几个问题问同一件事。
- `intent` 写这个问题的答案必须交代清楚什么。
- `lead` 留空或写一句话说明这份 FAQ 面向谁。
- 不要再另列 FAQ 字段——整篇就是 FAQ。""",
    generate_guide="""输出一份问答列表，**不要导语段落、不要论证过渡**：

```
# 标题

## 问题？

答案……[P12]

## 下一个问题？

答案……[P25]
```

- 每个答案 2–4 句，直接回答，第一句就给结论，不要铺垫。
- 答案之间彼此独立，不要出现「如前所述」「上文提到」这类互相引用。
- 不写结尾总结段。""",
)

BRAND = ContentTypeSpec(
    name="品牌介绍",
    section_word="板块",
    outline_guide="""这是一份品牌介绍，读者想知道「你们是谁、凭什么」。

- 规划 3–4 个**板块**，建议覆盖：我们是谁 / 我们解决什么问题 / 技术与能力凭据 /
  服务对象与实绩。资料支撑不足的板块直接不列。
- 板块标题用陈述句，克制、不夸张。
- `lead` 写开篇的品牌定位陈述应当覆盖什么。
- **不要 FAQ**——品牌介绍不是答疑页。""",
    generate_guide="""输出一份品牌介绍，**没有 FAQ 小节**：

```
# 标题

开篇定位陈述（2–3 句，说清这家公司是做什么的、为谁服务）。

## 板块标题

正文……[P12]
```

- 每个板块 150–300 字。
- 语气克制专业，用事实和实绩说话，不用「行业领先」「颠覆」这类自夸词——
  除非证据里确有第三方背书，那就引用它。
- 结尾一句话收束即可，不要号召性口号。""",
)

PRODUCT = ContentTypeSpec(
    name="产品介绍",
    section_word="能力模块",
    outline_guide="""这是一份产品介绍，读者想知道「它能做什么、怎么用、适合不适合我」。

- 规划 3–5 个**能力模块**，按功能而非按叙事组织。
- 若资料中有适用场景、服务对象、商业模式，单独成模块。
- 模块标题直接写能力名称，不要修辞。
- `lead` 写一句话价值主张应当覆盖什么。
- **不要 FAQ**。""",
    generate_guide="""输出一份产品介绍：

```
# 产品名 / 标题

一句话价值主张。

## 能力模块名

该模块解决什么、具体做什么……[P12]

- 要点一 [P25]
- 要点二 [P27]
```

- 每个模块先用 1–2 句说清它解决什么问题，再用要点列出具体能力。
- **允许使用要点列表**（这是产品介绍与 Blog 的主要区别），但每条要点也需引用。
- 不写导语式铺垫，不写结尾总结。""",
)

SPECS: dict[str, ContentTypeSpec] = {s.name: s for s in (BLOG, FAQ, BRAND, PRODUCT)}
CONTENT_TYPES = tuple(SPECS)


def get_spec(name: str) -> ContentTypeSpec:
    if name not in SPECS:
        raise ValueError(f"未知内容类型 {name!r}，可选：{CONTENT_TYPES}")
    return SPECS[name]
