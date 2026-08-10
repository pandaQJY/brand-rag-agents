# 语料来源与使用范围

本项目的两份语料都来自一家真实的 GEO 服务商，用于验证系统在真实材料上的行为——
合成语料无法暴露「设计稿式 PDF 三分之二内容藏在图里」「首页 FAQ 被单页预算静默截断」
这类只在真实数据上才出现的问题。

## 版本库里有什么

| 文件 | 内容 | 为何入库 |
|---|---|---|
| `pages.jsonl` | 品宣稿逐页解析结果（文字层 + 视觉转写） | 重建需 42 次视觉转写 API 调用，属付费步骤 |
| `chunks.jsonl` | 39 个检索单元 | 同上；检索、评测、写作链路都依赖它 |
| `faiss.index` / `faiss_ids.txt` | 知识库向量索引 | 由 chunks 重建，本地模型零成本，随附省一步 |
| `site_scan.json` | 官网结构扫描结果 | 只有统计量（h1 数、JSON-LD 覆盖数等），不含正文 |
| `preview.html` | 视觉转写人工抽检对照页 | 转写质量的核查凭据 |

## 版本库里没有什么

| 文件 | 为何排除 | 如何重建 |
|---|---|---|
| `source/brand_deck.pdf` | 该公司的原始设计稿，不转载 | 自行放入 `data/source/brand_deck.pdf` |
| `site_pages.jsonl`、`site_chunks.jsonl`、`site_faiss.*` | 官网正文属于该公司；且重建免费 | `python -m website.fetch && python -m website.chunk && python -m retrieval.build_index -c site`（约 1 分钟，零 API 成本） |
| `page_images/` | 8MB，仅重跑视觉转写时需要 | `python -m kb.parse` |

## 抓取行为

`website/fetch.py` 只抓取公开可访问的页面，单页间隔 0.6 秒（`config.CRAWL_DELAY`），
不绕过任何访问控制，不抓取需要留资或登录的内容——白皮书正文因此未被收录，
这一点本身也构成一条诊断结论（内容是好的，只是 AI 读不到）。

## 换成你自己的分析对象

系统不与任何特定品牌绑定。改 `.env` 三个变量即可：

```ini
SITE_BASE=https://your-target.com/
BRAND_NAME=你的品牌名
BRAND_ALIAS=英文名或产品名
```

Prompt 中的品牌名由 `core.llm.load_prompt` 自动注入，无需逐个文件替换。
知识库那一侧把 PDF 放到 `data/source/brand_deck.pdf` 后重跑 `kb.parse → kb.vision → kb.chunk`。
