from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from libgen import default_set_id, load_json, now_jst_iso, split_type_raw, write_json


BASE_TAGS = [
    {"tag_id": "format_chat", "label": "雑談系", "group": "format", "is_active": True},
    {"tag_id": "format_game", "label": "ゲーム系", "group": "format", "is_active": True},
    {"tag_id": "format_drawing", "label": "お絵描き系", "group": "format", "is_active": True},
    {"tag_id": "format_watchalong", "label": "同時視聴系", "group": "format", "is_active": True},
    {"tag_id": "format_sing", "label": "歌枠", "group": "format", "is_active": True},
    {"tag_id": "format_event", "label": "イベント系", "group": "format", "is_active": True},
    {"tag_id": "format_short", "label": "ショート", "group": "format", "is_active": True},
    {"tag_id": "attr_guerrilla", "label": "ゲリラ", "group": "attr", "is_active": True},
    {"tag_id": "tone_funny", "label": "笑い", "group": "tone", "is_active": True},
    {"tag_id": "tone_emotional", "label": "しんみり", "group": "tone", "is_active": True},
]

FUNNY_WORDS = {"爆笑", "カオス", "ボケ", "ゲス", "罵倒", "狂気", "おもろ", "笑い"}
EMOTIONAL_WORDS = {"泣", "しんみり", "葛藤", "寂し", "本音", "人生", "悩み", "つら"}


def add_tag(rows: dict[tuple[str, str], dict], video_id: str, tag_id: str, confidence: float, source: str) -> None:
    key = (video_id, tag_id)
    current = rows.get(key)
    if current and current["confidence"] >= confidence:
        return
    rows[key] = {
        "video_id": video_id,
        "tag_id": tag_id,
        "confidence": round(confidence, 3),
        "source": source,
    }


def build_tags(
    archives_path: Path,
    rules_path: Path,
    set_id: str,
    triggers_path: Path | None,
    unknown_output: Path | None,
) -> tuple[dict, dict]:
    archives = load_json(archives_path)["items"]
    rules = load_json(rules_path)
    split_delimiter = rules.get("split_delimiter", "＋")
    type_map: dict[str, list[str]] = rules.get("map", {})
    strict = bool(rules.get("strict", True))

    unknown_types: dict[str, int] = defaultdict(int)
    tag_rows: dict[tuple[str, str], dict] = {}

    for a in archives:
        video_id = a["video_id"]
        parts = split_type_raw(a.get("type_raw", ""), split_delimiter)
        for part in parts:
            mapped = type_map.get(part)
            if not mapped:
                unknown_types[part] += 1
                continue
            for tag_id in mapped:
                add_tag(tag_rows, video_id, tag_id, 1.0, "type_rule")

        if a.get("is_guerrilla"):
            add_tag(tag_rows, video_id, "attr_guerrilla", 1.0, "guerrilla_rule")

        summary = (a.get("summary_ai") or "").strip()
        if summary:
            if any(k in summary for k in FUNNY_WORDS):
                add_tag(tag_rows, video_id, "tone_funny", 0.7, "summary_rule")
            if any(k in summary for k in EMOTIONAL_WORDS):
                add_tag(tag_rows, video_id, "tone_emotional", 0.7, "summary_rule")

    tags = {t["tag_id"]: dict(t) for t in BASE_TAGS}

    if triggers_path:
        trigger_payload = load_json(triggers_path)
        for item in trigger_payload.get("items", []):
            trigger_id = item["trigger_id"]
            tags[trigger_id] = {
                "tag_id": trigger_id,
                "label": item["trigger"],
                "group": "motif",
                "is_active": True,
            }
            for video_id in item.get("video_ids", []):
                add_tag(tag_rows, video_id, trigger_id, 0.9, "trigger_rule")

    if unknown_types and strict:
        unknown_payload = {
            "set_id": set_id,
            "generated_at": now_jst_iso(),
            "unknown_types": [
                {"type_raw": t, "count": c} for t, c in sorted(unknown_types.items(), key=lambda x: (-x[1], x[0]))
            ],
        }
        unknown_path = unknown_output or Path("unknown_types.json")
        write_json(unknown_path, unknown_payload)
        raise ValueError(f"Unknown type_raw detected: {', '.join(sorted(unknown_types))}")

    tags_payload = {
        "version": 1,
        "generated_at": now_jst_iso(),
        "tags": sorted(tags.values(), key=lambda x: x["tag_id"]),
    }
    archive_tags_payload = {
        "version": 1,
        "generated_at": now_jst_iso(),
        "items": sorted(tag_rows.values(), key=lambda x: (x["video_id"], x["tag_id"])),
    }
    return tags_payload, archive_tags_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tags.json and archive_tags.json")
    parser.add_argument("--archives", default="public/data/archives.json")
    parser.add_argument("--rules", default="config/type_normalization_rules.json")
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--triggers", default=None, help="Optional trigger_candidates JSON path")
    parser.add_argument("--unknown-output", default=None)
    parser.add_argument("--tags-output", default="public/data/tags.json")
    parser.add_argument("--archive-tags-output", default="public/data/archive_tags.json")
    args = parser.parse_args()

    tags_payload, archive_tags_payload = build_tags(
        archives_path=Path(args.archives),
        rules_path=Path(args.rules),
        set_id=args.set_id,
        triggers_path=Path(args.triggers) if args.triggers else None,
        unknown_output=Path(args.unknown_output) if args.unknown_output else None,
    )
    write_json(Path(args.tags_output), tags_payload)
    write_json(Path(args.archive_tags_output), archive_tags_payload)
    print(
        f"Generated {args.tags_output} (tags={len(tags_payload['tags'])}), "
        f"{args.archive_tags_output} (rows={len(archive_tags_payload['items'])})"
    )


if __name__ == "__main__":
    main()
