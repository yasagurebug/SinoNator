from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from v2_common import build_auto_progress_points, load_json, load_pipeline_config, now_jst_iso, read_jsonl, write_json


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def resolve_count_from_factor(
    raw: Any,
    *,
    base_count: int,
    default_factor: float,
    min_value: int,
    max_value: int,
) -> int:
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"", "auto"}:
            factor = default_factor
        else:
            factor = float(s)
    elif raw is None:
        factor = default_factor
    else:
        factor = float(raw)

    if factor <= 0:
        value = 0
    else:
        value = int(round(base_count * factor))
    if factor > 0 and value < min_value:
        value = min_value
    return max(min_value if factor > 0 else 0, min(max_value, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step D: select concepts by split performance")
    parser.add_argument("--concepts", default="data/v2/concepts.json")
    parser.add_argument("--votes", default="data/v2/concept_votes.jsonl")
    parser.add_argument("--archives", default="data/v2/archives.json")
    parser.add_argument("--output", default="data/v2/selected_concepts.json")
    parser.add_argument("--config", default="config/v2_pipeline.json")
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    dcfg = cfg.get("step_d", {})
    min_yes_rate = float(dcfg.get("min_yes_rate", 0.05))
    max_yes_rate = float(dcfg.get("max_yes_rate", 0.95))
    max_unknown_rate = float(dcfg.get("max_unknown_rate", 0.60))
    redundant_jaccard_threshold = float(dcfg.get("redundant_jaccard_threshold", 0.82))
    rare_hook_keep_count_raw = dcfg.get("rare_hook_keep_count", "auto")
    rare_hook_max_support_raw = dcfg.get("rare_hook_max_support", "auto")
    rare_hook_min_yes_rate = float(dcfg.get("rare_hook_min_yes_rate", 0.001))
    rare_hook_max_yes_rate = float(dcfg.get("rare_hook_max_yes_rate", 0.12))
    rare_redundant_jaccard_threshold = float(dcfg.get("rare_redundant_jaccard_threshold", 0.95))
    core_priority_boost = float(dcfg.get("core_priority_boost", 0.25))

    concepts_payload = load_json(Path(args.concepts))
    concepts = list(concepts_payload.get("concepts", []))
    votes = read_jsonl(Path(args.votes))
    archives = load_json(Path(args.archives)).get("items", [])

    rec_ids = {str(x.get("video_id", "")) for x in archives if str(x.get("recommended_label", "")).strip()}
    total_videos = len(archives)
    overall_rec_rate = (len(rec_ids) / total_videos) if total_videos else 0.0
    rare_hook_keep_count = resolve_count_from_factor(
        rare_hook_keep_count_raw,
        base_count=total_videos,
        default_factor=0.14,
        min_value=0,
        max_value=400,
    )
    rare_hook_max_support = resolve_count_from_factor(
        rare_hook_max_support_raw,
        base_count=total_videos,
        default_factor=0.02,
        min_value=1,
        max_value=80,
    )

    concept_map = {str(c.get("concept_id", "")): c for c in concepts}
    bucket: dict[str, list[dict[str, Any]]] = {}
    for row in votes:
        cid = str(row.get("concept_id", ""))
        bucket.setdefault(cid, []).append(row)
    print(
        f"[StepD] start concepts={len(concepts)} vote_buckets={len(bucket)} videos={total_videos}",
        flush=True,
    )
    print(
        f"[StepD] resolved rare_hook_keep_count={rare_hook_keep_count} "
        f"rare_hook_max_support={rare_hook_max_support}",
        flush=True,
    )

    scored: list[dict[str, Any]] = []
    concept_stats: dict[str, dict[str, Any]] = {}
    score_points = build_auto_progress_points(len(bucket), reference_steps=total_videos)
    for i, (cid, rows) in enumerate(bucket.items(), start=1):
        yes_ids = {str(r.get("video_id", "")) for r in rows if str(r.get("label", "")) == "yes"}
        no_ids = {str(r.get("video_id", "")) for r in rows if str(r.get("label", "")) == "no"}
        unknown_ids = {str(r.get("video_id", "")) for r in rows if str(r.get("label", "")) == "unknown"}

        yes = len(yes_ids)
        no = len(no_ids)
        unknown = len(unknown_ids)
        total = yes + no + unknown
        known = yes + no
        if total == 0 or known == 0:
            continue

        yes_rate = yes / known
        unknown_rate = unknown / total
        p_yes = yes / known
        ent = binary_entropy(p_yes)
        gini = 1.0 - (p_yes * p_yes + (1.0 - p_yes) * (1.0 - p_yes))

        rec_yes = len(yes_ids & rec_ids)
        rec_yes_rate = (rec_yes / yes) if yes > 0 else 0.0
        rec_boost = max(0.0, rec_yes_rate - overall_rec_rate) * 0.2
        split_score = gini + (0.5 * ent) + rec_boost

        concept_stats[cid] = {
            "yes_ids": yes_ids,
            "yes_count": yes,
            "no_count": no,
            "unknown_count": unknown,
            "yes_rate": yes_rate,
            "unknown_rate": unknown_rate,
            "split_score": split_score,
            "rec_yes_rate": rec_yes_rate,
        }

        if yes_rate < min_yes_rate or yes_rate > max_yes_rate:
            continue
        if unknown_rate > max_unknown_rate:
            continue

        c = concept_map.get(cid, {})
        c_type = str(c.get("type", ""))
        lane = "core_type" if c_type == "core" else "core"
        adjusted_split = split_score + (core_priority_boost if c_type == "core" else 0.0)
        scored.append(
            {
                "concept_id": cid,
                "type": c_type,
                "name": c.get("name", ""),
                "definition": c.get("definition", ""),
                "seed_terms": c.get("seed_terms", []),
                "yes_count": yes,
                "no_count": no,
                "unknown_count": unknown,
                "yes_rate": round(yes_rate, 4),
                "unknown_rate": round(unknown_rate, 4),
                "split_score": round(adjusted_split, 6),
                "rec_yes_rate": round(rec_yes_rate, 4),
                "yes_video_ids": sorted(yes_ids),
                "selection_lane": lane,
            }
        )
        if i in score_points:
            print(f"[StepD] scoring {i}/{len(bucket)} (scored={len(scored)})", flush=True)

    scored.sort(key=lambda x: (-float(x["split_score"]), str(x.get("name", ""))))

    selected: list[dict[str, Any]] = []
    select_points = build_auto_progress_points(len(scored), reference_steps=total_videos)
    for i, row in enumerate(scored, start=1):
        ys = set(row["yes_video_ids"])
        redundant = False
        for s in selected:
            jy = jaccard(ys, set(s["yes_video_ids"]))
            if jy >= redundant_jaccard_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(row)
        if i in select_points:
            print(f"[StepD] selecting {i}/{len(scored)} (selected={len(selected)})", flush=True)

    rare_added = 0
    if rare_hook_keep_count > 0:
        selected_yes_sets = [set(s.get("yes_video_ids", [])) for s in selected]
        selected_ids = {str(s.get("concept_id", "")) for s in selected}
        rare_pool: list[tuple[float, int, str, dict[str, Any], set[str]]] = []
        for cid, stats in concept_stats.items():
            c = concept_map.get(cid, {})
            if str(c.get("type", "")) != "hook":
                continue
            if cid in selected_ids:
                continue
            yes_ids = set(stats["yes_ids"])
            if not yes_ids:
                continue
            support = int(c.get("support", len(yes_ids)))
            yes_rate = float(stats["yes_rate"])
            if support > rare_hook_max_support:
                continue
            if yes_rate < rare_hook_min_yes_rate or yes_rate > rare_hook_max_yes_rate:
                continue

            rarity = 1.0 / (1.0 + support)
            spice_score = (2.0 * rarity) + (0.4 * float(stats["split_score"])) + (0.2 * (1.0 - yes_rate))
            row = {
                "concept_id": cid,
                "type": c.get("type", ""),
                "name": c.get("name", ""),
                "definition": c.get("definition", ""),
                "seed_terms": c.get("seed_terms", []),
                "yes_count": int(stats["yes_count"]),
                "no_count": int(stats["no_count"]),
                "unknown_count": int(stats["unknown_count"]),
                "yes_rate": round(yes_rate, 4),
                "unknown_rate": round(float(stats["unknown_rate"]), 4),
                "split_score": round(float(stats["split_score"]), 6),
                "rec_yes_rate": round(float(stats["rec_yes_rate"]), 4),
                "yes_video_ids": sorted(yes_ids),
                "selection_lane": "spice",
                "spice_score": round(spice_score, 6),
            }
            rare_pool.append((spice_score, support, str(row["name"]), row, yes_ids))

        rare_pool.sort(key=lambda x: (-x[0], x[1], x[2]))
        for _, _, _, row, yes_ids in rare_pool:
            if rare_added >= rare_hook_keep_count:
                break
            redundant = False
            for ys in selected_yes_sets:
                if jaccard(yes_ids, ys) >= rare_redundant_jaccard_threshold:
                    redundant = True
                    break
            if redundant:
                continue
            selected.append(row)
            selected_yes_sets.append(yes_ids)
            rare_added += 1

        print(
            f"[StepD] spice selection added={rare_added} "
            f"(target={rare_hook_keep_count}, pool={len(rare_pool)})",
            flush=True,
        )

    payload = {
        "generated_at": now_jst_iso(),
        "source_concepts_path": args.concepts,
        "source_votes_path": args.votes,
        "thresholds": {
            "min_yes_rate": min_yes_rate,
            "max_yes_rate": max_yes_rate,
            "max_unknown_rate": max_unknown_rate,
            "redundant_jaccard_threshold": redundant_jaccard_threshold,
            "rare_hook_keep_count": rare_hook_keep_count,
            "rare_hook_max_support": rare_hook_max_support,
            "rare_hook_min_yes_rate": rare_hook_min_yes_rate,
            "rare_hook_max_yes_rate": rare_hook_max_yes_rate,
            "rare_redundant_jaccard_threshold": rare_redundant_jaccard_threshold,
            "core_priority_boost": core_priority_boost,
        },
        "all_scored_count": len(scored),
        "selected_count": len(selected),
        "selected_core_count": len([x for x in selected if str(x.get("selection_lane", "")).startswith("core")]),
        "selected_spice_count": len([x for x in selected if str(x.get("selection_lane", "")) == "spice"]),
        "items": selected,
    }
    write_json(Path(args.output), payload)
    print(
        f"Generated {args.output} "
        f"(scored={payload['all_scored_count']}, selected={payload['selected_count']})"
    )


if __name__ == "__main__":
    main()
