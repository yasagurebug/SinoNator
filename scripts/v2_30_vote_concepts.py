from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from v2_common import (
    build_auto_progress_points,
    call_openai_json,
    get_api_key,
    load_json,
    load_pipeline_config,
    normalize_text,
    now_jst_iso,
    read_jsonl,
    write_jsonl,
)


VOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "score": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["label", "score", "evidence"],
    "additionalProperties": False,
}


def concept_hash(concept: dict[str, Any]) -> str:
    raw = "||".join(
        [
            str(concept.get("concept_id", "")),
            str(concept.get("type", "")),
            str(concept.get("name", "")),
            str(concept.get("definition", "")),
            "|".join([str(x) for x in concept.get("seed_terms", [])]),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def video_hash(video: dict[str, Any]) -> str:
    raw = "||".join(
        [
            str(video.get("video_id", "")),
            str(video.get("title", "")),
            str(video.get("summary_ai", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def vote_hash(c_hash: str, v_hash: str) -> str:
    return hashlib.sha1(f"{c_hash}||{v_hash}".encode("utf-8")).hexdigest()


def keyword_match(text: str, terms: list[str]) -> tuple[str, float, str]:
    nt = normalize_text(text)
    for t in terms:
        ts = t.strip()
        if not ts:
            continue
        if normalize_text(ts) in nt:
            return "yes", 1.0, ts
    return "no", 0.0, ""


TYPE_SPLIT_RE = re.compile(r"[＋+／/,、・\s]+")


def type_match(type_raw: str, terms: list[str]) -> tuple[str, float, str]:
    raw = str(type_raw).strip()
    if not raw:
        return "no", 0.0, ""
    base_norm = normalize_text(raw)
    token_norms = {normalize_text(t) for t in TYPE_SPLIT_RE.split(raw) if str(t).strip()}
    token_norms.add(base_norm)
    for t in terms:
        ts = str(t).strip()
        if not ts:
            continue
        nt = normalize_text(ts)
        if nt in token_norms:
            return "yes", 1.0, ts
        if nt and nt in base_norm:
            return "yes", 1.0, ts
    return "no", 0.0, ""


def llm_vote_semantic(
    *,
    model: str,
    api_key: str,
    concept: dict[str, Any],
    title: str,
    summary: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict binary classifier for Japanese stream recommendation concepts. "
        "Classify against definition. Use 'unknown' if evidence is insufficient."
    )
    user_prompt = (
        f"Concept name: {concept.get('name','')}\n"
        f"Definition: {concept.get('definition','')}\n"
        f"Seed terms: {', '.join(concept.get('seed_terms', []))}\n\n"
        f"Title:\n{title}\n\n"
        f"Summary:\n{summary}\n\n"
        "Return label yes/no/unknown with confidence score and evidence snippet from input text."
    )
    out = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="semantic_vote",
        schema=VOTE_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=300,
        temperature=0.0,
    )
    label = str(out.get("label", "unknown")).lower()
    if label not in {"yes", "no", "unknown"}:
        label = "unknown"
    score = float(out.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    evidence = str(out.get("evidence", "")).strip()
    return {"label": label, "score": score, "evidence": evidence}


def mock_vote_semantic(concept: dict[str, Any], title: str, summary: str) -> dict[str, Any]:
    text = f"{title}\n{summary}".strip()
    if not summary.strip():
        return {"label": "unknown", "score": 0.0, "evidence": ""}
    label, score, ev = keyword_match(text, [str(x) for x in concept.get("seed_terms", [])])
    if label == "yes":
        return {"label": "yes", "score": 0.78, "evidence": ev}
    return {"label": "no", "score": 0.62, "evidence": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step C: concept x archive votes (incremental)")
    parser.add_argument("--archives", default="data/v2/archives.json")
    parser.add_argument("--concepts", default="data/v2/concepts.json")
    parser.add_argument("--output", default="data/v2/concept_votes.jsonl")
    parser.add_argument("--config", default="config/v2_pipeline.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    model = str(cfg.get("model", "gpt-5.2"))
    placeholder = str(cfg.get("openai_api_key_placeholder", "YOUR_OPENAI_API_KEY"))
    api_key = get_api_key(placeholder)

    archives = load_json(Path(args.archives))
    videos = list(archives.get("items", []))
    concepts_payload = load_json(Path(args.concepts))
    concepts = list(concepts_payload.get("concepts", []))
    if not videos or not concepts:
        raise ValueError("archives or concepts are empty")

    existing = read_jsonl(Path(args.output))
    existing_map: dict[tuple[str, str], dict] = {}
    for row in existing:
        key = (str(row.get("concept_id", "")), str(row.get("video_id", "")))
        existing_map[key] = row

    out_rows: list[dict] = []
    reused = 0
    updated = 0
    total = len(concepts) * len(videos)
    progress_points = build_auto_progress_points(total, reference_steps=len(videos))
    checkpoint_points = set(progress_points)
    checkpoint_points.add(total)
    processed = 0
    print(
        f"[StepC] start pairs={total} concepts={len(concepts)} videos={len(videos)} "
        f"mode={'mock' if args.mock else 'api'}",
        flush=True,
    )
    try:
        for concept in concepts:
            cid = str(concept.get("concept_id", "")).strip()
            c_type = str(concept.get("type", "")).strip()
            c_hash = concept_hash(concept)
            seed_terms = [str(x).strip() for x in concept.get("seed_terms", []) if str(x).strip()]
            for video in videos:
                vid = str(video.get("video_id", "")).strip()
                title = str(video.get("title", "")).strip()
                summary = str(video.get("summary_ai", "")).strip()
                type_raw = str(video.get("type_raw", "")).strip()
                v_hash = video_hash(video)
                vh = vote_hash(c_hash, v_hash)
                old = existing_map.get((cid, vid))
                action = "updated"
                if old and old.get("vote_hash") == vh:
                    out_rows.append(old)
                    reused += 1
                    action = "reused"
                else:
                    if c_type == "hook":
                        label, score, evidence = keyword_match(
                            f"{title}\n{summary}",
                            seed_terms or [concept.get("name", "")],
                        )
                        vote = {"label": label, "score": float(score), "evidence": evidence}
                    elif c_type == "core":
                        label, score, evidence = type_match(
                            type_raw,
                            seed_terms or [concept.get("name", "")],
                        )
                        vote = {"label": label, "score": float(score), "evidence": evidence}
                    else:
                        if args.mock:
                            vote = mock_vote_semantic(concept, title, summary)
                        else:
                            vote = llm_vote_semantic(
                                model=model,
                                api_key=api_key,
                                concept=concept,
                                title=title,
                                summary=summary,
                            )

                    out_rows.append(
                        {
                            "generated_at": now_jst_iso(),
                            "concept_id": cid,
                            "video_id": vid,
                            "label": vote["label"],
                            "score": round(float(vote["score"]), 4),
                            "evidence": vote["evidence"],
                            "vote_hash": vh,
                        }
                    )
                    updated += 1

                processed += 1
                if processed in checkpoint_points:
                    snapshot = sorted(out_rows, key=lambda r: (str(r.get("concept_id", "")), str(r.get("video_id", ""))))
                    write_jsonl(Path(args.output), snapshot)
                if processed in progress_points:
                    print(
                        f"[StepC] {processed}/{total} {cid}/{vid} {action} "
                        f"(reused={reused}, updated={updated})",
                        flush=True,
                    )
    except KeyboardInterrupt:
        if out_rows:
            snapshot = sorted(out_rows, key=lambda r: (str(r.get("concept_id", "")), str(r.get("video_id", ""))))
            write_jsonl(Path(args.output), snapshot)
        print(
            f"[StepC] interrupted after {processed}/{total} "
            f"(reused={reused}, updated={updated}, checkpoint_written={bool(out_rows)})",
            flush=True,
        )
        raise

    out_rows.sort(key=lambda r: (str(r.get("concept_id", "")), str(r.get("video_id", ""))))
    write_jsonl(Path(args.output), out_rows)
    print(
        f"Generated {args.output} "
        f"(total={len(out_rows)}, reused={reused}, updated={updated}, mock={args.mock})"
    )


if __name__ == "__main__":
    main()
