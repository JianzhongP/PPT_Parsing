"""离线合并人工审核结果并重生成 output.md。

使用场景：
- 已经跑过 main.py，目录内有 page_XXX.json
- 人工已在 review_results.json 填写修订结果
- 不希望再次调用模型，只想合并并导出最终 markdown
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _rewrite_markdown_image_paths(markdown: str, *, artifacts_dir: str | None, output_md_path: Path, project_root: Path) -> str:
    """将 markdown 中本地图片路径改写为相对 output.md 的路径。"""
    if not markdown:
        return ""

    img_pat = re.compile(r"(!\[[^\]]*\]\()\s*([^\)\s]+)([^\)]*)\)")
    out_dir = output_md_path.parent
    artifacts_path = (project_root / artifacts_dir) if artifacts_dir else None

    def _is_url(p: str) -> bool:
        return p.startswith("http://") or p.startswith("https://") or p.startswith("data:")

    def repl(m: re.Match) -> str:
        prefix, raw_path, suffix = m.group(1), m.group(2), m.group(3)
        path = raw_path.strip().strip('"').strip("'")
        if not path or _is_url(path):
            return m.group(0)

        candidate_paths: List[Path] = []
        p = Path(path)
        if p.is_absolute():
            candidate_paths.append(p)
        else:
            if artifacts_path:
                candidate_paths.append(artifacts_path / path)
            candidate_paths.append(project_root / path)

        target = None
        for c in candidate_paths:
            if c.exists():
                target = c
                break
        if target is None:
            return m.group(0)

        rel = os.path.relpath(str(target), str(out_dir)).replace(os.sep, "/")
        return f"{prefix}{rel}{suffix})"

    return img_pat.sub(repl, markdown)


def _selected_markdown(page_data: Dict[str, Any]) -> str:
    reviewed = page_data.get("reviewed_markdown_content")
    if isinstance(reviewed, str) and reviewed.strip():
        return reviewed
    return str(page_data.get("markdown_content", "") or "")


def _load_pages(output_dir: Path) -> Dict[int, Tuple[Path, Dict[str, Any]]]:
    pages: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    for page_file in sorted(output_dir.glob("page_*.json")):
        try:
            payload = json.loads(page_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        page_idx = payload.get("page_index")
        if not isinstance(page_idx, int):
            m = re.search(r"page_(\d{3})\.json$", page_file.name)
            if not m:
                continue
            page_idx = int(m.group(1))

        payload.setdefault("needs_review", False)
        payload.setdefault("review_status", "not_required")
        payload.setdefault("review_reasons", [])
        payload.setdefault("review_notes", "")
        payload.setdefault("reviewed_markdown_content", None)
        payload.setdefault("reviewed_summary", None)
        payload.setdefault("reviewed_elements", None)

        pages[page_idx] = (page_file, payload)
    return pages


def _load_review_results(review_results_path: Path) -> List[Dict[str, Any]]:
    if not review_results_path.exists():
        return []

    try:
        payload = json.loads(review_results_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _apply_reviews(pages: Dict[int, Tuple[Path, Dict[str, Any]]], reviews: List[Dict[str, Any]]) -> Tuple[int, int]:
    updated = 0
    skipped = 0

    for item in reviews:
        page_idx = item.get("page_index")
        if not isinstance(page_idx, int) or page_idx not in pages:
            skipped += 1
            continue

        _, page = pages[page_idx]

        status = str(item.get("review_status", "pending")).strip().lower()
        if status not in {"pending", "reviewed", "not_required"}:
            status = "pending"

        page["review_status"] = status
        page["review_notes"] = str(item.get("review_notes", "") or "")

        reviewed_summary = item.get("reviewed_summary")
        if isinstance(reviewed_summary, str) and reviewed_summary.strip():
            page["reviewed_summary"] = reviewed_summary.strip()

        reviewed_md = item.get("reviewed_markdown_content")
        if isinstance(reviewed_md, str) and reviewed_md.strip():
            page["reviewed_markdown_content"] = reviewed_md.strip()

        reviewed_elements = item.get("reviewed_elements")
        if isinstance(reviewed_elements, dict) and reviewed_elements:
            page["reviewed_elements"] = reviewed_elements

        if status == "reviewed":
            page["needs_review"] = False

        updated += 1

    return updated, skipped


def _render_markdown(pages: Dict[int, Tuple[Path, Dict[str, Any]]], output_md_path: Path, project_root: Path) -> str:
    lines: List[str] = ["# PPT解析结果", ""]

    for page_idx in sorted(pages.keys()):
        _, page = pages[page_idx]
        page_no = page_idx + 1
        section_title = str(page.get("section_title", "") or "").strip()
        header = f"## 第 {page_no} 页" + (f"：{section_title}" if section_title else "")

        md = _selected_markdown(page)
        md_stripped = md.lstrip()
        has_page_header = bool(re.match(r"^##\s*(第\s*\d+\s*页|页面\s*\d+)\b", md_stripped))

        artifacts_dir = None
        slice_reference = page.get("slice_reference")
        if isinstance(slice_reference, dict):
            artifacts_dir = slice_reference.get("artifacts_dir")

        md = _rewrite_markdown_image_paths(
            md,
            artifacts_dir=artifacts_dir,
            output_md_path=output_md_path,
            project_root=project_root,
        )

        if not has_page_header:
            lines.extend([header, ""])

        lines.extend([md, "", "---", ""])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并人工审核结果并重生成 output.md")
    parser.add_argument("--output-dir", default="ppt_parsing_output", help="包含 page_*.json 的目录")
    parser.add_argument("--review-results", default=None, help="review_results.json 路径，默认 <output-dir>/review_results.json")
    parser.add_argument("--output-md", default=None, help="输出 markdown 路径，默认 <output-dir>/output.md")
    parser.add_argument("--write-page-json", action="store_true", help="将合并后的审核字段写回 page_*.json")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    review_results_path = Path(args.review_results).resolve() if args.review_results else output_dir / "review_results.json"
    output_md_path = Path(args.output_md).resolve() if args.output_md else output_dir / "output.md"

    pages = _load_pages(output_dir)
    if not pages:
        raise RuntimeError(f"未找到可用页面结果: {output_dir}/page_*.json")

    reviews = _load_review_results(review_results_path)
    updated, skipped = _apply_reviews(pages, reviews)

    if args.write_page_json:
        for _, (page_file, page_data) in pages.items():
            page_file.write_text(json.dumps(page_data, ensure_ascii=False, indent=2), encoding="utf-8")

    project_root = Path(__file__).parent.resolve()
    merged_md = _render_markdown(pages, output_md_path, project_root)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(merged_md, encoding="utf-8")

    reviewed_pages = sum(1 for _, page in pages.values() if str(page.get("review_status", "")).lower() == "reviewed")

    print("=" * 70)
    print("[merge_review] 合并完成")
    print(f"- 页面总数: {len(pages)}")
    print(f"- 读取审核记录: {len(reviews)}")
    print(f"- 应用成功: {updated}")
    print(f"- 跳过记录: {skipped}")
    print(f"- 已审核页面: {reviewed_pages}")
    print(f"- 输出文件: {output_md_path}")
    if args.write_page_json:
        print(f"- 已回写页面JSON: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
