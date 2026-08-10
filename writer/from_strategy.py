"""把内容策略的建议，变成带引用的成稿。

**这是诊断闭环的最后一环。**

Agent 链路本身走到内容策略就停了：它产出「应该补一篇 FAQ，回答『如何让 AI 推荐
我们的产品』」，然后把这句话交给人。而写作链路需要的正是一个主题。
两者对接，才构成完整的一条内容运营流水线：

    抓官网 → 诊断 → 客户会怎么问 → 该补什么内容 → 补出来

真正让这个衔接有价值的不是「省了一次复制粘贴」，而是**成稿的事实边界**：
写作链路只能引用品宣稿知识库里的内容，且要过四层半幻觉拦截。
因此这里产出的不是「AI 编的一篇稿子」，而是**只由品牌方自己已公开的事实
组装而成、每句话都能溯源到页码的初稿**——它可以直接进入人工润色流程，
而不必先做一遍事实核对。

## 类型映射

内容策略按 GEO 口径分类（FAQ / Blog / 案例页 / 产品说明 / 元数据修改），
写作链路按文体分类（官网Blog / 品牌介绍 / 产品介绍 / FAQ）。两者不是一一对应：

- 「元数据修改」根本不是一篇文章，是改 meta 标签的工程任务，不进写作链路；
- 「案例页」映射到官网Blog，但**知识库里没有客户案例的细节**，
  生成的必然是泛泛之谈——因此默认跳过并说明原因，而不是硬写一篇空话。

**能写的才写，写不了的说清为什么写不了。** 把不可写的项也塞进流水线，
只会得到一堆需要人逐句删改的稿子，那比不生成更糟。

用法：
    python -m writer.from_strategy --plan            # 只看能写哪些，不调模型
    python -m writer.from_strategy --top 2           # 写最高优先级的 2 篇
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from core.config import EXAMPLE_DIR
from writer.content_types import CONTENT_TYPES
from writer.generate import Article, generate
from writer.literal import check as literal_check
from writer.literal import summarize as literal_summarize
from writer.outline import plan
from writer.retrieve import build_evidence
from writer.verify import verify

# 内容策略的类型 → 写作链路的文体。值为 None 表示不进写作链路。
TYPE_MAP: dict[str, str | None] = {
    "FAQ": "FAQ",
    "Blog": "官网Blog",
    "博客": "官网Blog",
    "产品说明": "产品介绍",
    "产品介绍": "产品介绍",
    "品牌介绍": "品牌介绍",
    "白皮书": "官网Blog",  # 长文形态与 Blog 一致，只是篇幅更长
    "案例页": None,  # 知识库无客户案例细节，硬写只会得到空话
    "案例": None,
    "元数据修改": None,  # 是工程任务，不是文章
    "元数据": None,
}

SKIP_REASON = {
    "案例页": "知识库中没有客户案例的细节事实，生成的只会是泛泛之谈——"
    "这类内容必须由真实交付材料支撑，不该由模型补全",
    "案例": "同上：缺少可引用的案例事实",
    "元数据修改": "这是改 meta 标签的工程任务，不是一篇文章",
    "元数据": "这是工程任务，不是一篇文章",
}

PRIORITY_ORDER = {"高优先级": 0, "中优先级": 1, "低优先级": 2}

# 内容策略输出的建议行形如：**FAQ｜如何让 AI 推荐我们的产品？**
# 分隔符在实测中出现过全角｜与半角|两种，一并接住。
_REC_RE = re.compile(r"^\*\*(?P<type>[^｜|*]+)[｜|](?P<title>.+?)\*\*$")


@dataclass
class Recommendation:
    """内容策略给出的一条建议。"""

    raw_type: str
    title: str
    priority: str
    content_type: str | None = None  # 映射后的文体；None 表示不可写
    skip_reason: str = ""

    @property
    def writable(self) -> bool:
        return self.content_type is not None

    @property
    def sort_key(self) -> int:
        return PRIORITY_ORDER.get(self.priority, 9)


@dataclass
class Draft:
    """一条建议写成的初稿，连同它的核查结果。"""

    rec: Recommendation
    article: Article
    literal: dict = field(default_factory=dict)
    verify_pass_rate: float = 0.0
    risky: int = 0


def parse_recommendations(strategy_text: str) -> list[Recommendation]:
    """从内容策略的行式输出里抽出建议条目。

    只认「加粗的 类型｜标题」这一行，其下的「为什么做 / 预期效果」是给人看的
    理由，不参与写作——主题已经由标题给全了，把理由一并塞进去反而会让
    大纲规划器围着理由展开，写出一篇解释「我们为什么要写这篇」的文章。
    """
    recs: list[Recommendation] = []
    seen: set[tuple[str, str]] = set()
    priority = ""

    for raw in strategy_text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            priority = line.lstrip("# ").strip()
            continue
        # 收口环节会把同一批建议再复述一遍（标题写作 **高优先级** 而非 ## ），
        # 因此也认这种写法，否则复述段落里的条目会顶着上一个 ## 标题的优先级。
        if (bold := re.fullmatch(r"\*\*(高优先级|中优先级|低优先级|不建议现在做)\*\*", line)):
            priority = bold.group(1)
            continue
        if priority == "不建议现在做":
            continue

        # 只剥列表符号，不能连 ** 一起剥——lstrip("-*· ") 会把加粗标记也吃掉，
        # 导致下面的正则永远匹配不上。
        m = _REC_RE.match(re.sub(r"^[-·]\s*", "", line))
        if not m:
            continue

        raw_type = m.group("type").strip()
        title = m.group("title").strip()
        if (raw_type, title) in seen:  # 收口复述的同一条，不重复计数
            continue
        seen.add((raw_type, title))

        mapped = TYPE_MAP.get(raw_type, "官网Blog")  # 未知类型按 Blog 处理
        recs.append(
            Recommendation(
                raw_type=raw_type,
                title=title,
                priority=priority,
                content_type=mapped,
                skip_reason=SKIP_REASON.get(raw_type, "") if mapped is None else "",
            )
        )
    return sorted(recs, key=lambda r: r.sort_key)


def write_draft(rec: Recommendation) -> Draft:
    """按一条建议跑完整条写作链路，含第 3.5 层与第 4 层核查。"""
    if not rec.writable:
        raise ValueError(f"该建议不进写作链路：{rec.skip_reason}")

    outline = plan(rec.title, rec.content_type)
    pack = build_evidence(outline)
    article = generate(pack, rec.content_type)
    report = verify(article, pack)
    claims = literal_check(article.markdown)

    return Draft(
        rec=rec,
        article=article,
        literal=literal_summarize(claims),
        verify_pass_rate=report.pass_rate,
        risky=len(report.risky),
    )


def load_strategy(path=None) -> str:
    """读取一份已保存的内容策略产出。

    默认找 examples/agents/ 下最近一次实跑记录，便于不重跑 Agent 链路
    就能演示这一环。
    """
    from pathlib import Path

    if path:
        return Path(path).read_text(encoding="utf-8")

    candidates = sorted((EXAMPLE_DIR / "agents").glob("*.md"))
    if not candidates:
        raise FileNotFoundError(
            "未找到内容策略产出。先运行 python -m apps.cli \"官网该补充什么内容\"，"
            "或用 --strategy 指定文件。"
        )
    return candidates[-1].read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="把内容策略建议写成带引用的初稿")
    ap.add_argument("--strategy", help="内容策略产出的文件路径，默认取 examples/agents/ 最新一份")
    ap.add_argument("--plan", action="store_true", help="只列出能写哪些，不调模型")
    ap.add_argument("--top", type=int, default=1, help="按优先级写前 N 篇")
    ap.add_argument("--type", choices=CONTENT_TYPES, help="只写指定文体的建议")
    args = ap.parse_args()

    recs = parse_recommendations(load_strategy(args.strategy))
    if not recs:
        print("未从内容策略中解析出建议条目。检查输入文件是否为内容策略的产出。")
        return

    writable = [r for r in recs if r.writable]
    if args.type:
        writable = [r for r in writable if r.content_type == args.type]

    print(f"共解析出 {len(recs)} 条建议，其中 {len(writable)} 条可进写作链路：\n")
    for r in recs:
        mark = "✓" if r.writable else "—"
        line = f" {mark} [{r.priority}] {r.raw_type}｜{r.title}"
        print(line if r.writable else f"{line}\n     跳过：{r.skip_reason}")

    if args.plan:
        return

    for rec in writable[: args.top]:
        print("\n" + "═" * 72)
        print(f"写作：{rec.content_type}《{rec.title}》")
        print("═" * 72)
        draft = write_draft(rec)
        a = draft.article
        print(f"\n{a.markdown}\n")
        print("─" * 72)
        print(
            f"字数 {a.stats['字数']}　引用 {a.stats['引用总数']} 处 / 覆盖 "
            f"{a.stats['引用页数']} 页　捏造引用 {len(a.invalid_cites)} 处"
        )
        print(
            f"第 3.5 层数字校验：{draft.literal}　"
            f"第 4 层核查通过率：{draft.verify_pass_rate:.0%}（{draft.risky} 条待处理）"
        )


if __name__ == "__main__":
    main()
