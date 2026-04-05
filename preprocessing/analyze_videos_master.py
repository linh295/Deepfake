from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


VALID_SPLITS = ("train", "val", "test")


def normalize_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_split(value: object) -> str:
    text = normalize_text(value).lower()
    return text if text in VALID_SPLITS else "missing"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze train/val/test distribution from videos_master.csv."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root_dir / "artifacts" / "videos_master.csv",
        help="Path to videos_master.csv",
    )
    return parser.parse_args()


def load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"video_id", "category", "split"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError("Manifest is missing required columns: " + ", ".join(missing))
        return list(reader)


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def _render_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    all_rows = [headers] + str_rows
    widths = [max(len(row[idx]) for row in all_rows) for idx in range(len(headers))]

    def fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def build_report(rows: List[Dict[str, str]]) -> str:
    total_videos = len(rows)
    overall_counts: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_split_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        category = normalize_text(row.get("category")) or "missing"
        split = normalize_split(row.get("split"))
        overall_counts[split] += 1
        category_totals[category] += 1
        category_split_counts[category][split] += 1

    split_order = list(VALID_SPLITS)
    if overall_counts.get("missing", 0) > 0:
        split_order.append("missing")

    summary_rows = [
        [split, overall_counts.get(split, 0), _format_percent(overall_counts.get(split, 0), total_videos)]
        for split in split_order
    ]

    category_rows = []
    for category in sorted(category_totals):
        total = category_totals[category]
        row = [category, total]
        for split in split_order:
            count = category_split_counts[category].get(split, 0)
            row.extend([count, _format_percent(count, total)])
        category_rows.append(row)

    category_headers = ["category", "total"]
    for split in split_order:
        category_headers.extend([f"{split}_count", f"{split}_pct"])

    parts = [
        f"Manifest: {len(rows)} videos",
        "",
        "Overall split distribution",
        _render_table(["split", "count", "pct"], summary_rows),
        "",
        "Per-category split distribution",
        _render_table(category_headers, category_rows),
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    rows = load_manifest_rows(manifest_path)
    print(build_report(rows))


if __name__ == "__main__":
    main()
