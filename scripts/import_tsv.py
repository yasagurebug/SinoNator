from __future__ import annotations

import argparse
from pathlib import Path

from libgen import now_jst_iso, parse_circle_bool, parse_start_time, read_source_tsv, write_json


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
    for row in rows:
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

    payload = {
        "generated_at": now_jst_iso(),
        "source_tsv": tsv_path.name,
        "items": items,
        "stats": {
            "total_rows": len(rows),
            "valid_items": len(items),
            "dropped_rows": bad_rows,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate public/data/archives.json from TSV")
    parser.add_argument(
        "--tsv",
        default="specs/しののめにこ配信データベース - nicoライブリスト.tsv",
        help="Input TSV path",
    )
    parser.add_argument(
        "--output",
        default="public/data/archives.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    output_path = Path(args.output)
    payload = build_archives(tsv_path)
    write_json(output_path, payload)
    stats = payload["stats"]
    print(
        f"Generated {output_path} "
        f"(rows={stats['total_rows']}, valid={stats['valid_items']}, dropped={stats['dropped_rows']})"
    )


if __name__ == "__main__":
    main()
