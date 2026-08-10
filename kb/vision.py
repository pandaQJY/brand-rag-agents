"""视觉通道：把每页渲染图交给视觉模型转写为结构化文本。

产物写回 data/pages.jsonl 的 vision_text 字段。已转写的页默认跳过，
因此中途失败可直接重跑续做，不会重复付费。

用法：
    python -m ingest.vision                # 转写所有尚未转写的页
    python -m ingest.vision --pages 3 8 20 # 只转写指定页（先验证质量再全量）
    python -m ingest.vision --force        # 强制重转已完成的页
    python -m ingest.vision --workers 4    # 并发数
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import DATA_DIR
from kb.models import PageDoc
from kb.parse import load_pages, save_pages
from core.llm import load_prompt, vision

_lock = threading.Lock()


def _strip_fence(text: str) -> str:
    """剥掉模型有时自作主张加上的 ``` 围栏，避免污染索引文本。"""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[-1] if "\n" in t else ""
    if "```" in t:
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _dedupe_lines(text: str) -> str:
    """逐小节去除重复条目。

    模型对「同一名称只列一次」的遵守不稳定——同一媒体既作截图标签又出现在
    logo 墙时常被列两次。去重是确定性任务，交给代码比反复调 Prompt 可靠。
    """
    out: list[str] = []
    seen: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):  # 进入新小节，去重表清空
            seen.clear()
            out.append(line)
            continue
        key = stripped.lstrip("-*·0123456789. ").rstrip("　 ")
        if not key:
            out.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)

    return "\n".join(out)


def _postprocess(text: str) -> str:
    return _dedupe_lines(_strip_fence(text))


def transcribe_page(page: PageDoc) -> tuple[PageDoc, dict]:
    """转写一页。文字层作为校对参照一并送入，降低专名识别错误。"""
    prompt = load_prompt(
        "kb/vision_transcribe",
        text_layer=(page.text_layer or "（该页无可提取文字）"),
    )
    text, usage = vision(DATA_DIR / page.image_path, prompt)
    page.vision_text = _postprocess(text)
    return page, usage


def run(pages: list[PageDoc], targets: list[int], workers: int) -> dict:
    """并发转写指定页，逐页落盘以支持断点续传。"""
    by_no = {p.page_no: p for p in pages}
    todo = [by_no[n] for n in targets]
    total_usage = {"in": 0, "out": 0}
    done = 0

    def _work(pg: PageDoc):
        return transcribe_page(pg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, pg): pg for pg in todo}
        for fut in as_completed(futures):
            pg = futures[fut]
            try:
                _, usage = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {pg.cite_id}  失败：{str(e)[:120]}")
                continue

            with _lock:
                done += 1
                total_usage["in"] += usage["in"]
                total_usage["out"] += usage["out"]
                save_pages(pages)  # 逐页落盘：中断后可续做
                lines = pg.vision_text.count("\n") + 1
                print(f"  {pg.cite_id}  {len(pg.vision_text):>4}字 / {lines:>2}行  "
                      f"(in {usage['in']} out {usage['out']})  [{done}/{len(todo)}]")

    return total_usage


def main() -> None:
    ap = argparse.ArgumentParser(description="视觉通道批量转写")
    ap.add_argument("--pages", type=int, nargs="*", help="只处理这些页码（默认全部未完成页）")
    ap.add_argument("--force", action="store_true", help="重转已完成的页")
    ap.add_argument("--workers", type=int, default=4, help="并发数，默认 4")
    args = ap.parse_args()

    pages = load_pages()

    if args.pages:
        targets = args.pages
    elif args.force:
        targets = [p.page_no for p in pages]
    else:
        targets = [p.page_no for p in pages if not p.vision_text]

    if not targets:
        print("所有页均已转写。用 --force 可强制重转。")
        return

    print(f"待转写 {len(targets)} 页，并发 {args.workers}：")
    usage = run(pages, targets, args.workers)

    filled = sum(1 for p in pages if p.vision_text)
    print(f"\n完成。已转写 {filled}/{len(pages)} 页")
    print(f"token 用量：输入 {usage['in']:,}  输出 {usage['out']:,}")


if __name__ == "__main__":
    main()
