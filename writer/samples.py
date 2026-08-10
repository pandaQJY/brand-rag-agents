"""生成四种内容类型的示例，每份含输入、输出与事实核查明细。

产物写入 examples/writing/。每份样例都完整记录：主题、内容类型、
检索到的证据、生成正文、逐条核查结果——读的人无需运行代码即可判断系统行为，
也便于在改动 Prompt 后比对前后差异。

用法：
    python -m writer.samples              # 生成全部四份
    python -m writer.samples --only FAQ   # 只生成一份
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import BASE_DIR, BRAND_ALIAS, BRAND_NAME
from writer.content_types import CONTENT_TYPES
from writer.generate import generate
from writer.outline import plan
from writer.retrieve import build_evidence
from writer.literal import check as literal_check
from writer.literal import summarize as literal_summarize
from writer.verify import LABELS, verify

# 示例放在项目根的 examples/ 而不是埋在 data/ 里——
# 它是给人看的记录，不是运行期产物。
SAMPLE_DIR = BASE_DIR / "examples" / "writing"

# (内容类型, 主题, 输出文件名)
CASES = [
    ("官网Blog", "为什么中国出海品牌需要进行GEO优化", "01_官网Blog.md"),
    ("品牌介绍", f"{BRAND_NAME}是一家什么样的公司", "02_品牌介绍.md"),
    ("产品介绍", f"{BRAND_ALIAS or BRAND_NAME} GEO 产品能力", "03_产品介绍.md"),
    ("FAQ", f"{BRAND_NAME}的GEO服务", "04_FAQ.md"),
]


def build_one(content_type: str, topic: str, filename: str) -> dict:
    outline = plan(topic, content_type)
    pack = build_evidence(outline)
    article = generate(pack, content_type)
    report = verify(article, pack)
    nums = literal_check(article.markdown)
    ns = literal_summarize(nums)

    st = article.stats
    lines = [
        f"# 示例：{content_type}",
        "",
        "## 输入",
        "",
        f"- **主题**：{topic}",
        f"- **内容类型**：{content_type}",
        "",
        "## 检索到的证据",
        "",
    ]
    for i, se in enumerate(pack.sections, 1):
        ids = " ".join(h.chunk.chunk_id for h in se.hits)
        lines.append(f"{i}. **{se.section.heading}**")
        for q in se.section.queries:
            lines.append(f"   - 查询：`{q}`")
        lines.append(f"   - 证据：{ids}")
    lines += [
        "",
        f"去重后共 **{len(pack.chunks)}** 份证据。",
        "",
        "## 输出",
        "",
        "---",
        "",
        article.markdown,
        "",
        "---",
        "",
        "## 自动核查结果",
        "",
        f"- 第 3 层（引用编号有效性）："
        + (f"⚠ 捏造编号 {article.invalid_cites}" if article.invalid_cites else "✅ 通过，无捏造编号"),
        f"- 第 3.5 层（数字确定性校验）：{len(nums)} 个量化表述 → "
        f"逐字命中 {ns['exact']}、取整 {ns['approx']}、"
        + (f"**编造 {ns['missing']}** ⚠" if ns["missing"] else "编造 0 ✅"),
        f"- 第 4 层（事实核查）：核查 {len(report.claims)} 条事实性陈述，"
        f"通过率 **{report.pass_rate:.0%}**",
        "",
        f"正文 {st['字数']} 字 / {st['句子数']} 句，其中 {st['带引用句']} 句带引用；"
        f"引用 {st['引用总数']} 处、覆盖 {st['引用页数']} 页。",
        "",
        "| 事实性陈述 | 引用 | 判定 | 判据 |",
        "|---|---|---|---|",
    ]
    for c in report.claims:
        name = LABELS[c.verdict][0]
        mark = "✅" if c.verdict == "supported" else "⚠️"
        cited = "/".join(c.cited) if c.cited else "—"
        text = c.text.replace("|", "\\|")[:60]
        reason = c.reason.replace("|", "\\|")[:50]
        lines.append(f"| {text} | {cited} | {mark} {name} | {reason} |")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DIR / filename).write_text("\n".join(lines), encoding="utf-8")

    return {
        "type": content_type,
        "file": filename,
        "words": st["字数"],
        "cites": st["引用总数"],
        "claims": len(report.claims),
        "pass": report.pass_rate,
        "num_bad": ns["missing"],
        "invalid": len(article.invalid_cites),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="生成示例交付物")
    ap.add_argument("--only", choices=CONTENT_TYPES, help="只生成指定类型")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or c[0] == args.only]
    results = []
    for ct, topic, fn in cases:
        print(f"生成 {ct} …", flush=True)
        results.append(build_one(ct, topic, fn))

    print(f"\n{'类型':<10}{'字数':>6}{'引用':>6}{'声明':>6}{'通过率':>8}{'捏造编号':>9}{'编造数字':>9}")
    print("─" * 56)
    for r in results:
        print(f"{r['type']:<10}{r['words']:>6}{r['cites']:>6}{r['claims']:>6}"
              f"{r['pass']:>7.0%}{r['invalid']:>8}{r['num_bad']:>9}")
    print(f"\n已写入 {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
