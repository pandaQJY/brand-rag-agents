"""事实核查：幻觉拦截的第 4 层。

第 3 层只能验证「引用编号存不存在」，验证不了两件更危险的事：
    1. 编号真实，但那一页根本没说这件事（miscited / unsupported）
    2. 句子写了事实却根本没标引用（第 3 层完全看不见）
本模块把文章里的事实性陈述逐条抽出，对照全部证据判定。

输出沿用行式格式而非 JSON——与大纲模块同因：本项目实测该模型在嵌套
结构上不可靠，而记录块式文本解析确定且零失败。

用法：
    python -m writer.verify "为什么中国出海品牌需要进行GEO优化"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from core.llm import chat, load_prompt
from writer.generate import Article, generate
from writer.literal import check as literal_check
from writer.literal import summarize as literal_summarize
from writer.outline import plan
from writer.retrieve import EvidencePack, build_evidence

VERDICTS = ("supported", "miscited", "unsupported", "contradicted")
RISKY = ("unsupported", "contradicted", "miscited")

# 判定 → (中文名, 终端色)
LABELS = {
    "supported": ("有据支持", "\033[32m"),
    "miscited": ("引用指错页", "\033[36m"),
    "unsupported": ("无证据支撑", "\033[33m"),
    "contradicted": ("与证据矛盾", "\033[31m"),
}
RESET = "\033[0m"


@dataclass
class Claim:
    text: str
    cited: list[str] = field(default_factory=list)
    verdict: str = "unsupported"
    reason: str = ""

    @property
    def is_risky(self) -> bool:
        return self.verdict in RISKY


@dataclass
class VerifyReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def risky(self) -> list[Claim]:
        return [c for c in self.claims if c.is_risky]

    @property
    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for c in self.claims:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    @property
    def pass_rate(self) -> float:
        return (self.counts["supported"] / len(self.claims)) if self.claims else 0.0


def parse_report(text: str) -> VerifyReport:
    """解析记录块式核查结果。"""
    report = VerifyReport()
    current: Claim | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("==="):
            if current:
                report.claims.append(current)
                current = None
            continue

        tag, sep, value = line.partition(":")
        if not sep:
            continue
        tag, value = tag.strip().upper(), value.strip()

        if tag == "CLAIM":
            if current:
                report.claims.append(current)
            current = Claim(text=value)
        elif current is None:
            continue
        elif tag == "CITED":
            if value and value not in ("无", "None", "-"):
                current.cited = [p.strip() for p in value.replace("，", ",").split(",") if p.strip()]
        elif tag == "VERDICT":
            v = value.lower()
            current.verdict = v if v in VERDICTS else "unsupported"
        elif tag == "REASON":
            current.reason = value

    if current:
        report.claims.append(current)
    return report


def verify(article: Article, pack: EvidencePack, provider: str | None = None) -> VerifyReport:
    prompt = load_prompt("writer/verify", article=article.markdown, evidence=pack.full_context())
    raw = chat([{"role": "user", "content": prompt}], provider=provider, task="verify", max_tokens=4096)
    return parse_report(raw)


def render(report: VerifyReport, color: bool = True) -> str:
    """把核查结果渲染成可读报告。"""
    lines = []
    counts = report.counts
    lines.append(f"核查 {len(report.claims)} 条事实性陈述，通过率 {report.pass_rate:.0%}")
    for v in VERDICTS:
        name, c = LABELS[v]
        tint, reset = (c, RESET) if color else ("", "")
        lines.append(f"  {tint}{name:<12}{counts[v]:>3}{reset}")

    if report.risky:
        lines.append("\n需要处理的陈述：")
        for i, cl in enumerate(report.risky, 1):
            name, c = LABELS[cl.verdict]
            tint, reset = (c, RESET) if color else ("", "")
            cited = "/".join(cl.cited) if cl.cited else "未标引用"
            lines.append(f"\n{i}. {tint}[{name}]{reset} 引用：{cited}")
            lines.append(f"   「{cl.text}」")
            lines.append(f"   判据：{cl.reason}")
    else:
        lines.append("\n未发现无支撑或矛盾的陈述。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成并核查文章")
    ap.add_argument("topic")
    ap.add_argument("--type", default="官网Blog")
    ap.add_argument("--provider")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    print("规划大纲…")
    outline = plan(args.topic, args.type, args.provider)
    print("分节检索…")
    pack = build_evidence(outline)
    print(f"生成中（{len(pack.chunks)} 份证据）…")
    article = generate(pack, args.type, args.provider)
    print("事实核查中…\n")
    report = verify(article, pack, args.provider)

    print("═" * 70)
    print("第 3 层｜引用有效性：", end="")
    print(f"捏造引用 {article.invalid_cites}" if article.invalid_cites else "通过，无捏造编号")

    nums = literal_check(article.markdown)
    ns = literal_summarize(nums)
    print(f"第 3.5 层｜数字确定性校验：{len(nums)} 个量化表述 → "
          f"逐字命中 {ns['exact']} / 取整 {ns['approx']} / 编造 {ns['missing']}")
    for c in nums:
        if c.status == "missing":
            print(f"    ✗ 编造「{c.value}」 {c.note}")
        elif c.status == "approx":
            print(f"    ⚠ 取整「{c.value}」 {c.note}")
    print("═" * 70)
    print(render(report, color=not args.no_color))


if __name__ == "__main__":
    main()
