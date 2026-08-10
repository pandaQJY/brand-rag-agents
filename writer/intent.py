"""意图解析：把一句自然语言指令拆成「内容类型 + 写作主题」。

真实使用中，用户给的是一整句指令：

    请基于品宣稿，生成一篇官网Blog，主题是："为什么中国出海品牌需要进行GEO优化？"

因此输入不应拆成「主题输入框 + 类型下拉框」两个控件——用户会把整句粘进来。

解析策略是**规则优先、模型兜底**：
四种内容类型的关键词有限且确定，规则命中率高、零成本、零延迟；
只有当规则识别不出类型或主题时，才调用一次模型。

用法：
    python -m writer.intent "请生成一篇官网Blog，主题是：为什么出海品牌需要GEO"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from core.llm import chat
from writer.content_types import CONTENT_TYPES

# 内容类型关键词。顺序有意义：先匹配更具体的词，避免「产品介绍」被「介绍」抢走。
_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("FAQ", ("faq", "常见问题", "问答", "答疑")),
    ("产品介绍", ("产品介绍", "产品说明", "产品页")),
    ("品牌介绍", ("品牌介绍", "公司介绍", "关于我们", "品牌故事")),
    ("官网Blog", ("blog", "博客", "文章", "推文", "软文")),
]

# 主题的常见引导语。中英文标点都要覆盖——用户从 Word 粘贴时全角居多。
_TOPIC_PATTERNS = [
    r"主题是[：:]\s*[\"“”'‘’《]?(.+?)[\"“”'‘’》]?\s*$",
    r"主题[：:]\s*[\"“”'‘’《]?(.+?)[\"“”'‘’》]?\s*$",
    r"关于[\"“”'‘’《]?(.+?)[\"“”'‘’》]?\s*(?:的|这个)?(?:文章|内容|blog|博客)?\s*$",
    r"[\"“”《]([^\"“”《》]{6,})[\"“”》]",  # 整句中被引号括起的长片段
]

# 指令性前缀，主题里不该保留
_NOISE = re.compile(
    r"(请|帮我|麻烦)?(基于|根据|依据)?[^，,。]*?(品宣|PDF|资料|материал)[^，,。]*?[，,]?"
    r"|(请|帮我|麻烦)?(生成|写|撰写|输出)(一篇|一份|一个)?",
)


@dataclass
class WriteRequest:
    topic: str
    content_type: str
    type_source: str = "规则"  # 规则 | 模型 | 默认
    topic_source: str = "规则"

    @property
    def parsed_by_model(self) -> bool:
        return "模型" in (self.type_source, self.topic_source)


def _match_type(text: str) -> str | None:
    low = text.lower()
    for ctype, keys in _TYPE_RULES:
        if any(k in low for k in keys):
            return ctype
    return None


def _match_topic(text: str) -> str | None:
    for pat in _TOPIC_PATTERNS:
        m = re.search(pat, text.strip(), re.I)
        if m:
            topic = m.group(1).strip(" 　\"'“”‘’《》。？?！!，,")
            # 「关于X的东西/的内容」这类口语尾巴不属于主题
            topic = re.sub(r"的(东西|内容|事|事情|玩意儿?)$", "", topic).strip()
            if len(topic) >= 4:
                return topic
    return None


def _strip_instruction(text: str) -> str:
    """去掉「请基于…生成一篇…」这类指令外壳，剩下的当主题。"""
    cleaned = _NOISE.sub("", text)
    for ctype, keys in _TYPE_RULES:
        for k in keys:
            cleaned = re.sub(re.escape(k), "", cleaned, flags=re.I)
    return cleaned.strip(" 　,，。：:、\"'“”‘’")


_LLM_PROMPT = """从下面这句写作指令中提取两项信息，严格按格式输出两行，不要任何解释：

TYPE: <只能是 {types} 之一；若指令未说明，填 官网Blog>
TOPIC: <写作主题本身，去掉「请基于…生成…」这类指令外壳，不要加引号>

指令：{text}"""


def _ask_model(text: str) -> tuple[str | None, str | None]:
    raw = chat(
        [{"role": "user", "content": _LLM_PROMPT.format(types=" / ".join(CONTENT_TYPES), text=text)}],
        task="intent",
        max_tokens=256,
    )
    ctype = topic = None
    for line in raw.splitlines():
        tag, sep, val = line.partition(":")
        if not sep:
            continue
        tag, val = tag.strip().upper(), val.strip().strip("\"'“”‘’《》")
        if tag == "TYPE" and val in CONTENT_TYPES:
            ctype = val
        elif tag == "TOPIC" and val:
            topic = val
    return ctype, topic


def parse_request(text: str, allow_model: bool = True) -> WriteRequest:
    """解析一句写作指令。规则优先，规则不中时才调模型。"""
    text = text.strip()
    if not text:
        raise ValueError("输入为空")

    ctype = _match_type(text)
    topic = _match_topic(text)
    type_src = "规则" if ctype else ""
    topic_src = "规则" if topic else ""

    # 规则没提取出主题：先试着剥掉指令外壳
    if not topic:
        stripped = _strip_instruction(text)
        if len(stripped) >= 6:
            topic, topic_src = stripped, "规则"

    # 仍有缺口才调模型——这是唯一会产生成本与延迟的分支
    if allow_model and (not ctype or not topic):
        m_type, m_topic = _ask_model(text)
        if not ctype and m_type:
            ctype, type_src = m_type, "模型"
        if not topic and m_topic:
            topic, topic_src = m_topic, "模型"

    if not ctype:
        ctype, type_src = "官网Blog", "默认"
    if not topic:
        topic, topic_src = text, "默认"

    return WriteRequest(topic=topic, content_type=ctype, type_source=type_src, topic_source=topic_src)


def main() -> None:
    ap = argparse.ArgumentParser(description="解析写作指令")
    ap.add_argument("text")
    ap.add_argument("--no-model", action="store_true", help="只用规则，不调模型")
    args = ap.parse_args()

    req = parse_request(args.text, allow_model=not args.no_model)
    print(f"内容类型：{req.content_type}   （来源：{req.type_source}）")
    print(f"写作主题：{req.topic}   （来源：{req.topic_source}）")


if __name__ == "__main__":
    main()
