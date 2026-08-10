"""第 3.5 层：数字与量化表述的确定性校验。

为什么单设这一层：

1. **数字是幻觉最危险的载体。** A/B 实测中，deepseek-v3 把证据里的
   「61.66 亿」写成「80 亿」，并凭空造出「300%」——句子读起来专业、
   还挂着真实的引用编号，人眼极难发现。

2. **LLM 核查器对数字不可靠。** 实测同一篇文章两次核查结果不同（1 条 vs 5 条），
   存在采样波动；且它曾漏过 GEO 缩写编造。

3. **这件事本不需要模型。** 数字要么在语料里逐字出现，要么没有。
   `in corpus` 一次判定，零成本、零波动、无漏报——
   凡能用确定性方法裁决的，就不该交给模型。

判定分三档：
    exact   —— 逐字命中，可信
    approx  —— 整数部分命中但写法不同（如证据 61.66 亿 → 文章 61 亿），
               属于四舍五入，需人工确认是否可接受
    missing —— 语料中完全没有，判定为编造

用法：
    python -m writer.literal article.md

注意：传入的应是**文章正文**。若对 examples/*.md 直接运行，
其中的核查报告（"核查 10 条陈述"之类）也会被当作正文校验，产生假阳性。
样例文件请用 --body 只取正文部分。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from kb.chunk import load_chunks

# 引用标记里的数字（[P08-09]）不是事实性数字，先剔除
_CITE = re.compile(r"\[P\d{2}(?:-\d{2})?\]")
# Markdown 标记与小节序号也不算
_MD = re.compile(r"^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s", re.M)

# 量化表述：百分比 / 倍数 / 量级 / 带单位的计数
_NUM = re.compile(
    r"(\d+(?:\.\d+)?\s*%)"                                  # 30%  17.6%
    r"|(\d+(?:\.\d+)?\s*倍)"                                 # 15倍
    r"|(\d+(?:\.\d+)?\s*(?:万亿|千万|百万|亿|万)(?:次|人|元|美元)?)"  # 61.66亿  3.45亿次  5万亿
    r"|(\d+\s*\+)"                                          # 20+
    r"|(\d+(?:\.\d+)?\s*(?:家|个|条|款|年|个月|天|小时))"        # 200家  两个月
)

# 这些数字属于行文而非数据，不必校验
_IGNORE = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}

SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


@dataclass
class NumberClaim:
    value: str  # 文中写法，如 "80亿"
    sentence: str
    status: str  # exact | approx | missing
    note: str = ""

    @property
    def is_risky(self) -> bool:
        return self.status != "exact"


def _normalize(v: str) -> str:
    return re.sub(r"\s+", "", v)


def _appears_literally(val: str, corpus: str) -> bool:
    """字面命中必须带数字边界。

    直接用 `val in corpus` 会把「4万」判为命中——因为它是语料中「834万」的子串。
    左侧需排除数字与小数点，右侧需排除数字。
    """
    return re.search(r"(?<![\d.])" + re.escape(val) + r"(?!\d)", corpus) is not None


# 数值 + 量级单位。用于把「61.66亿」解析成 (61.66, "亿")，做数值比较而非字符串包含。
_NUMVAL = re.compile(r"(\d+(?:\.\d+)?)\s*(%|倍|万亿|千万|百万|亿|万)?\s*(次|人|元|美元|家|个|条|款)?")


def _parse(v: str) -> tuple[float, str] | None:
    """把「1.8万美元」解析成 (1.8, "万美元")。

    单位必须含计数词：语料有「5万」但无「万美元」，
    若单位只取「万」，编造的「超1.8万美元」会被「5万」蒙混过关。
    """
    m = _NUMVAL.match(v)
    if not m:
        return None
    try:
        return float(m.group(1)), (m.group(2) or "") + (m.group(3) or "")
    except ValueError:
        return None


def _corpus_values(corpus: str) -> set[tuple[float, str]]:
    """把语料里所有量化表述解析成 (数值, 复合单位) 集合。"""
    out: set[tuple[float, str]] = set()
    for m in _NUMVAL.finditer(corpus):
        try:
            out.add((float(m.group(1)), (m.group(2) or "") + (m.group(3) or "")))
        except ValueError:
            continue
    return out


# 下界修饰词：「超3亿」在真值 3.45亿 时是成立的保守表述，不是编造。
_LOWER = ("超过", "超", "突破", "逾", "多于", "大于", "不低于", "高于", "达", "上万", "余")
_UPPER = ("不足", "低于", "少于", "不到", "仅", "以内")


def _bound_of(prefix: str, suffix: str) -> str:
    """判断数值前后的修饰词，决定这是精确声明还是上/下界声明。"""
    if any(w in prefix for w in _UPPER):
        return "upper"
    if any(w in prefix for w in _LOWER) or suffix.startswith("余"):
        return "lower"
    return "exact"


def _bound_ok(v: float, unit: str, bound: str, pool: set[tuple[float, str]]) -> float | None:
    """下界/上界声明是否被语料支持。

    注意这只校验「语料里存在这样量级的数」，不校验归属对象——
    「豆包超3亿」与「ChatGPT超3亿」在本层无法区分，那是第 4 层的职责。
    """
    same = [cv for cv, cu in pool if cu == unit]
    if not same:
        return None
    if bound == "lower":
        # 只接受同一数量级内的支撑：文中「超4万」被语料某个「4000万」撞上，
        # 数值上虽成立，实际多半是不相干的数据。取最接近的那个。
        ok = [cv for cv in same if v <= cv <= v * 10]
        return min(ok) if ok else None
    if bound == "upper":
        ok = [cv for cv in same if cv <= v]
        return max(ok) if ok else None
    return None


def _is_rounding_of(v: float, unit: str, pool: set[tuple[float, str]]) -> float | None:
    """文中数值是否为语料中某个同单位数值的四舍五入/截断。

    仅接受 2% 以内的偏差——61.66亿→61亿 属于取整，61.66亿→80亿 不是。
    """
    for cv, cu in pool:
        if cu != unit or cv == 0:
            continue
        if abs(cv - v) / abs(cv) <= 0.02:
            return cv
    return None


def extract_numbers(markdown: str) -> list[tuple[str, str, str, str]]:
    """抽出 (数字写法, 所在句子, 前缀, 后缀)。前后缀用于判定上下界修饰词。"""
    text = _CITE.sub("", markdown)
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = _MD.sub("", line).strip()
        if not line:
            continue
        for sent in SENT_SPLIT.split(line):
            sent = sent.strip()
            if not sent:
                continue
            for m in _NUM.finditer(sent):
                val = _normalize(next(g for g in m.groups() if g))
                if val and val not in _IGNORE:
                    out.append((val, sent, sent[max(0, m.start() - 5):m.start()], sent[m.end():m.end() + 3]))
    return out


def check(markdown: str, corpus: str | None = None) -> list[NumberClaim]:
    """把文中每个量化表述与语料逐字比对。"""
    if corpus is None:
        corpus = "\n".join(c.index_text() for c in load_chunks())
    flat = _normalize(corpus)
    pool = _corpus_values(flat)

    claims: list[NumberClaim] = []
    seen: set[tuple[str, str]] = set()
    for val, sent, prefix, suffix in extract_numbers(markdown):
        key = (val, sent[:30])
        if key in seen:
            continue
        seen.add(key)

        if _appears_literally(val, flat):
            claims.append(NumberClaim(val, sent, "exact"))
            continue

        parsed = _parse(val)
        if parsed:
            v, unit = parsed
            bound = _bound_of(prefix, suffix)
            if bound != "exact":
                src = _bound_ok(v, unit, bound, pool)
                if src is not None:
                    word = "下界" if bound == "lower" else "上界"
                    claims.append(NumberClaim(
                        val, sent, "exact",
                        f"{word}表述，语料原值 {src:g}{unit}，该表述成立"))
                    continue
            src = _is_rounding_of(v, unit, pool)
            if src is not None:
                claims.append(
                    NumberClaim(val, sent, "approx",
                                f"语料原值为 {src:g}{unit}，文中写作「{val}」，属取整或改写")
                )
                continue
        claims.append(NumberClaim(val, sent, "missing", "语料中不存在该数值"))
    return claims


def summarize(claims: list[NumberClaim]) -> dict:
    out = {"exact": 0, "approx": 0, "missing": 0}
    for c in claims:
        out[c.status] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="数字确定性校验（第 3.5 层）")
    ap.add_argument("path", help="文章 Markdown 文件")
    ap.add_argument("--body", action="store_true",
                    help="输入是 examples/*.md 时，只截取「## 输出」与「## 自动核查结果」之间的正文")
    args = ap.parse_args()

    md = Path(args.path).read_text(encoding="utf-8")
    if args.body and "## 输出" in md:
        md = md.split("## 输出", 1)[1].split("## 自动核查结果", 1)[0]
    claims = check(md)
    s = summarize(claims)

    print(f"抽出 {len(claims)} 个量化表述：逐字命中 {s['exact']} / 近似 {s['approx']} / 编造 {s['missing']}\n")
    for c in claims:
        if c.status == "exact":
            continue
        tag = "⚠ 近似" if c.status == "approx" else "✗ 编造"
        print(f"{tag}  {c.value}")
        print(f"      句子：{c.sentence[:70]}")
        print(f"      判据：{c.note}\n")
    if not any(c.is_risky for c in claims):
        print("全部逐字命中语料。")


if __name__ == "__main__":
    main()
