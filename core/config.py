"""全局路径与参数。所有模块从这里取配置，不散落魔法值。

本模块被所有入口最先导入，因此把必须早于 torch / faiss 加载的环境变量
设置也放在这里。

配置按三条链路分区：知识库构建（PDF）、站点抓取、上下文预算。
"""

import os
from pathlib import Path

# faiss-cpu 与 torch 各自捆绑了一份 OpenMP 运行时。同一进程内先用 torch
# 编码、再调 faiss 检索时，两份 libomp 冲突会导致段错误（无 traceback，
# 进程直接退出）。允许重复加载是 Intel OpenMP 官方给出的规避方式。
# 必须在 import torch / faiss 之前设置，否则无效。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 单线程可进一步避免线程池争用；语料仅数十条，无性能影响。
os.environ.setdefault("OMP_NUM_THREADS", "1")
# tokenizers 在 fork 后会打印一段并行警告。检索路径本就不依赖其并行，
# 显式关掉，免得每次运行都在正常输出里混进四行无关提示。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# core/ 的上一级即仓库根目录
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
PAGE_IMAGE_DIR = DATA_DIR / "page_images"
PROMPT_DIR = BASE_DIR / "prompts"
EXAMPLE_DIR = BASE_DIR / "examples"

# ── 知识库链路：品宣 PDF → 可检索、可引用的语料 ──────────────

SOURCE_PDF = SOURCE_DIR / "brand_deck.pdf"
PAGES_JSONL = DATA_DIR / "pages.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
KB_INDEX = DATA_DIR / "faiss.index"
KB_IDMAP = DATA_DIR / "faiss_ids.txt"

# --- 页面渲染 ---
# 长边像素。1600 在视觉模型的识别质量与 token 成本之间取平衡；
# 原始页面为 960x540(16:9)，放大约 1.67 倍，中文小字仍清晰。
# 1600x900 ≈ 1.44M 像素，Qwen-VL 约计 1800 token/张，全 42 页约 7.6 万 token。
RENDER_MAX_EDGE = 1600

# 渲染格式。PNG 文字最锐利但单页约 1.8MB，42 页 15MB，base64 后送 API 过重；
# JPEG q=90 在同等分辨率下约 200KB，OCR 质量无可感差异。
RENDER_FORMAT = "jpg"
JPEG_QUALITY = 90

# --- 解析阈值 ---
# 文字层低于此字符数的页，判定为「文字层近空」，其内容完全依赖视觉通道。
MIN_TEXT_CHARS = 30

# 判定两个文本块属于同一视觉行的最小垂直重叠比例。
ROW_OVERLAP_RATIO = 0.5

# 标题候选须落在页面上部此比例区域内。
TITLE_TOP_RATIO = 0.45

# ── 站点链路：官网抓取 → Agent 可通读的语料 ──────────────────

SITE_PAGES_JSONL = DATA_DIR / "site_pages.jsonl"
SCAN_JSON = DATA_DIR / "site_scan.json"
# 跨页重复的模板文案（公告条、页脚、全局导航）。从各页正文中剔除后单独存放，
# 供 Agent 通读一次——它是品牌表达的一部分，值得看见，但不该在 52 页里各看一遍。
BOILERPLATE_JSON = DATA_DIR / "site_boilerplate.json"

# 站点语料的检索单元与索引。与知识库各自建索引：两者的引用锚点不同
# （知识库锚到页码，站点锚到 URL），混在一个索引里无法区分证据出处。
SITE_CHUNKS_JSONL = DATA_DIR / "site_chunks.jsonl"
SITE_INDEX = DATA_DIR / "site_faiss.index"
SITE_IDMAP = DATA_DIR / "site_faiss_ids.txt"

# 抓取目标
SITE_BASE = os.getenv("SITE_BASE", "https://www.wanxitech.cn/")
CRAWL_MAX_PAGES = 120
CRAWL_DELAY = 0.6  # 秒，礼貌延迟


def site_domain(base: str = "") -> str:
    """从站点根地址取注册域，用于判定「站内链接」。

    取末两段（wanxitech.cn），而非整个 netloc（www.wanxitech.cn）——
    否则 blog.example.com 这类子域会被判成站外，抓取在第一跳就断。
    """
    from urllib.parse import urlparse

    host = urlparse(base or SITE_BASE).netloc
    return ".".join(host.split(".")[-2:]) if host.count(".") >= 1 else host


# ── 品牌标识 ────────────────────────────────────────────────
#
# Prompt 里不写死品牌名：换一个分析对象时应当只改配置，不改 Prompt。
# core.llm.load_prompt 会把这两个值自动注入所有 Prompt 的 {brand}/{brand_alias}。
BRAND_NAME = os.getenv("BRAND_NAME", "万悉科技")
BRAND_ALIAS = os.getenv("BRAND_ALIAS", "Trendee")  # 英文名 / 产品名，可留空

# ── 上下文预算 ──────────────────────────────────────────────
#
# 只有一个绝对数：总预算。其余都是它的占比。
# 初版是三个互不相干的绝对字数（单页 3000 / 样本 1000 / 每板块 5 篇），
# 每个都是照着这个站点试出来的，换站即失效，且失效得静默——
# 首页 FAQ 区块起始于第 3,016 字，单页额度 3,000，整块被无声截掉，
# 诊断据此判出「官网缺少 GEO 与 SEO 区别的 FAQ」，而它就在首页上。
SITE_DIGEST_BUDGET = 30_000

# 分配顺序：全站大纲 → 身份页正文 → 板块样本（余额）。
# 大纲吃头一份，因为**覆盖比深度重要**：全站 h1–h3 大纲约 18k 字就能让 52 页
# 全部露面，比只覆盖 8 页正文的样本还便宜。「官网有没有讲 X」这类问题的答案
# 在标题里，不在正文里——覆盖该由站点自己的目录保证，不该靠抽样碰运气。
OUTLINE_SHARE = 0.62
IDENTITY_SHARE = 0.15

# 每个板块取几篇正文样本。这是方法论选择，不是预算选择：
# 覆盖已由大纲兜住，样本只用来判断「这个站是怎么说话的」，几篇代表作足矣。
SECTION_SAMPLES = 3


def ensure_dirs() -> None:
    """创建运行期需要的目录。"""
    for d in (DATA_DIR, SOURCE_DIR, PAGE_IMAGE_DIR, PROMPT_DIR, EXAMPLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
