from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from v2_common import build_auto_progress_points, load_json, now_jst_iso, write_json


JST = timezone(timedelta(hours=9))

REQUIRED_COLUMNS = [
    "video_id",
    "video_url",
    "title",
    "start_time",
    "タイプ",
    "ゲリラ",
    "おすすめ",
    "要約（AI利用）",
]


def read_source_tsv(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if len(rows) < 3:
        raise ValueError("TSV rows are too short")

    header = rows[1]
    data_start = 3 if rows[2] == header else 2

    items: list[dict[str, str]] = []
    for row in rows[data_start:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        data = {header[i]: padded[i].strip() for i in range(len(header))}
        items.append(data)
    return items


def parse_start_time(raw: str) -> str:
    value = raw.strip()
    dt = datetime.strptime(value, "%Y/%m/%d %H:%M:%S")
    return dt.replace(tzinfo=JST).isoformat(timespec="seconds")


def parse_circle_bool(raw: str) -> bool:
    return raw.strip() == "〇"


def build_archives(tsv_path: Path) -> dict:
    rows = read_source_tsv(tsv_path)
    if not rows:
        raise ValueError("No data rows found in TSV")

    for col in REQUIRED_COLUMNS:
        if col not in rows[0]:
            raise ValueError(f"Missing TSV column: {col}")

    items = []
    bad_rows = 0
    seen_ids: set[str] = set()
    points = build_auto_progress_points(len(rows), reference_steps=len(rows))
    for i, row in enumerate(rows, start=1):
        video_id = row["video_id"].strip()
        video_url = row["video_url"].strip()
        title = row["title"].strip()
        start_time_raw = row["start_time"].strip()
        if not video_id or not video_url or not title or not start_time_raw:
            bad_rows += 1
            continue
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        summary = row["要約（AI利用）"].strip()
        item = {
            "video_id": video_id,
            "video_url": video_url,
            "title": title,
            "start_time": parse_start_time(start_time_raw),
            "type_raw": row["タイプ"].strip(),
            "is_guerrilla": parse_circle_bool(row["ゲリラ"]),
            "recommended_label": row["おすすめ"].strip(),
            "summary_ai": summary,
            "has_summary": bool(summary),
        }
        items.append(item)
        if i in points:
            print(
                f"[Step0] {i}/{len(rows)} parsed "
                f"(valid={len(items)}, dropped={bad_rows})",
                flush=True,
            )

    return {
        "generated_at": now_jst_iso(),
        "source_tsv": tsv_path.name,
        "items": items,
        "stats": {
            "total_rows": len(rows),
            "valid_items": len(items),
            "dropped_rows": bad_rows,
        },
    }


def row_hash(item: dict) -> str:
    raw = "||".join(
        [
            str(item.get("video_id", "")),
            str(item.get("video_url", "")),
            str(item.get("title", "")),
            str(item.get("start_time", "")),
            str(item.get("type_raw", "")),
            str(item.get("is_guerrilla", "")),
            str(item.get("recommended_label", "")),
            str(item.get("summary_ai", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def same_items(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    am = {str(x.get("video_id", "")): row_hash(x) for x in a}
    bm = {str(x.get("video_id", "")): row_hash(x) for x in b}
    return am == bm


def main() -> None:
    parser = argparse.ArgumentParser(description="v2: Generate data/v2/archives.json from TSV")
    parser.add_argument(
        "--tsv",
        default="specs/しののめにこ配信データベース - nicoライブリスト.tsv",
        help="Input TSV path",
    )
    parser.add_argument(
        "--output",
        default="data/v2/archives.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    output_path = Path(args.output)
    payload = build_archives(tsv_path)
    if output_path.exists():
        prev = load_json(output_path)
        prev_items = list(prev.get("items", []))
        if same_items(prev_items, payload["items"]):
            stats = payload["stats"]
            print(
                f"Skipped {output_path} "
                f"(unchanged rows={stats['total_rows']}, valid={stats['valid_items']})"
            )
            return
    write_json(output_path, payload)
    stats = payload["stats"]
    print(
        f"Generated {output_path} "
        f"(rows={stats['total_rows']}, valid={stats['valid_items']}, dropped={stats['dropped_rows']})"
    )


if __name__ == "__main__":
    main()
