from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any

from v2_common import (
    build_auto_progress_points,
    call_openai_json,
    get_api_key,
    load_json,
    load_pipeline_config,
    now_jst_iso,
    stable_id,
    write_json,
)


QUESTION_STYLE_VERSION = "v4_pref_question_keep_quotes_2026_02_15"

ALLOWED_ENDINGS = (
    "は好きですか？",
    "に興味がありますか？",
    "は見たいですか？",
    "はよく見ますか？",
    "は気になりますか？",
    "は楽しめそうですか？",
    "は今の気分に合いますか？",
    "は見てみたいですか？",
)

BLOCKED_PHRASES = (
    "この配信",
    "この動画",
    "このおすすめ",
    "役立ちそう",
    "差別",
    "暴力",
    "攻撃",
    "誹謗中傷",
)

QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}


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


def concept_source_hash(concept: dict[str, Any]) -> str:
    raw = "||".join(
        [
            QUESTION_STYLE_VERSION,
            str(concept.get("concept_id", "")),
            str(concept.get("name", "")),
            str(concept.get("definition", "")),
            "|".join([str(x) for x in concept.get("seed_terms", [])]),
            str(concept.get("split_score", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def mock_question_text(concept: dict[str, Any]) -> str:
    topic = normalize_topic(str(concept.get("name", "")))
    return build_preference_question(topic, str(concept.get("concept_id", "")))


def normalize_topic(text: str) -> str:
    t = " ".join(text.strip().split())
    t = t.replace("\n", " ").strip()
    t = t.strip("「」『』\"'")
    t = re.sub(r"^この(配信|動画|おすすめ|質問)\s*(は|が)?", "", t)
    t = re.sub(r"(ですか|ますか)[？?]?$", "", t)
    t = t.strip("、。!?！？ ")
    if len(t) > 28:
        t = t[:28].rstrip("、。!?！？ ")
    return t


def build_preference_question(topic: str, concept_id: str) -> str:
    seed = hashlib.sha1(concept_id.encode("utf-8")).hexdigest()
    use_interest = int(seed[:2], 16) % 2 == 1
    if use_interest:
        if "配信" in topic:
            return f"{topic}に興味がありますか？"
        return f"{topic}に興味がありますか？"
    if "配信" in topic:
        return f"{topic}は好きですか？"
    return f"{topic}といった配信は好きですか？"


def sanitize_question_text(topic_text: str, fallback: str, concept_id: str) -> str:
    topic = normalize_topic(topic_text)
    if not topic:
        return fallback
    blocked = [
        "差別",
        "暴力",
        "攻撃",
        "誹謗中傷",
        "この配信",
        "この動画",
        "このおすすめ",
        "役立ちそう",
    ]
    if any(b in topic for b in blocked):
        return fallback
    q = build_preference_question(topic, concept_id)
    if len(q) > 48:
        return fallback
    return q


def normalize_generated_question(text: str) -> str:
    t = " ".join(text.strip().split())
    t = t.replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    if t and not t.endswith(("？", "?")):
        t += "？"
    t = t.replace("?", "？")
    return t


def is_valid_question_text(text: str) -> bool:
    if not text:
        return False
    if len(text) > 56:
        return False
    if any(b in text for b in BLOCKED_PHRASES):
        return False
    if text.count("？") > 1:
        return False
    return any(text.endswith(end) for end in ALLOWED_ENDINGS)


def llm_question_text(
    *,
    model: str,
    api_key: str,
    concept: dict[str, Any],
) -> str:
    system_prompt = (
        "You write one concise Japanese yes/no preference question for livestream recommendation UI. "
        "Return exactly one sentence."
    )
    endings_text = " / ".join(ALLOWED_ENDINGS)
    user_prompt = (
        f"Concept name: {concept.get('name','')}\n"
        f"Type: {concept.get('type','')}\n"
        f"Definition: {concept.get('definition','')}\n"
        f"Seed terms: {', '.join(concept.get('seed_terms', []))}\n"
        "Constraints:\n"
        "- Must be natural Japanese for end users.\n"
        "- Ask user preference/interest only.\n"
        "- Must end with ONE of these endings exactly:\n"
        f"  {endings_text}\n"
        "- Do not use: この配信 / この動画 / このおすすめ / 役立ちそう\n"
        "- One sentence only, <= 56 chars.\n"
    )
    out = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="question_text",
        schema=QUESTION_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=96,
        temperature=0.4,
    )
    return str(out.get("text", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step E: generate UI question text")
    parser.add_argument("--selected", default="data/v2/selected_concepts.json")
    parser.add_argument("--output", default="data/v2/questions_v2.json")
    parser.add_argument("--config", default="config/v2_pipeline.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    model = str(cfg.get("model", "gpt-5.2"))
    placeholder = str(cfg.get("openai_api_key_placeholder", "YOUR_OPENAI_API_KEY"))
    api_key = get_api_key(placeholder)
    step_e_cfg = cfg.get("step_e", {})
    llm_retry_attempts = max(1, int(step_e_cfg.get("llm_retry_attempts", 3)))

    selected_payload = load_json(Path(args.selected))
    concepts = list(selected_payload.get("items", []))
    variable_count_raw = step_e_cfg.get("variable_count", "auto")
    variable_count = resolve_count_from_factor(
        variable_count_raw,
        base_count=max(1, len(concepts)),
        default_factor=1.0,
        min_value=1,
        max_value=1000,
    )
    concepts = concepts[: max(1, variable_count)]
    print(f"[StepE] resolved variable_count={len(concepts)}", flush=True)

    out_path = Path(args.output)
    existing_map: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        prev = load_json(out_path)
        for row in prev.get("items", []):
            existing_map[str(row.get("concept_id", ""))] = row

    rows: list[dict[str, Any]] = []
    reused = 0
    updated = 0
    fallback_used = 0
    retry_success = 0
    total = len(concepts)
    progress_points = build_auto_progress_points(total, reference_steps=total)
    print(f"[StepE] start concepts={total} mode={'mock' if args.mock else 'api'}", flush=True)
    for i, c in enumerate(concepts, start=1):
        cid = str(c.get("concept_id", ""))
        c_type = str(c.get("type", "")).strip().lower()
        if c_type not in {"core", "hook", "semantic"}:
            c_type = "hook"
        sh = concept_source_hash(c)
        old = existing_map.get(cid)
        action = "updated"
        if old and str(old.get("source_hash", "")) == sh:
            reused_row = dict(old)
            reused_row["question_type"] = c_type
            rows.append(reused_row)
            reused += 1
            action = "reused"
        else:
            fallback = mock_question_text(c)
            if args.mock:
                text = fallback
            else:
                text = ""
                for attempt in range(1, llm_retry_attempts + 1):
                    raw = llm_question_text(model=model, api_key=api_key, concept=c)
                    candidate = normalize_generated_question(raw)
                    if is_valid_question_text(candidate):
                        text = candidate
                        if attempt > 1:
                            retry_success += 1
                        break
                if not text:
                    text = fallback
                    fallback_used += 1
            rows.append(
                {
                    "question_id": f"qv2_{stable_id('q', cid)[:14]}",
                    "concept_id": cid,
                    "question_type": c_type,
                    "text": text,
                    "priority": i,
                    "source_hash": sh,
                }
            )
            updated += 1
        if i in progress_points:
            print(
                f"[StepE] {i}/{total} {cid} {action} "
                f"(reused={reused}, updated={updated})",
                flush=True,
            )

    payload = {
        "generated_at": now_jst_iso(),
        "source_selected_path": args.selected,
        "item_count": len(rows),
        "items": rows,
        "meta": {
            "reused": reused,
            "updated": updated,
            "mock": args.mock,
            "llm_retry_attempts": llm_retry_attempts,
            "retry_success": retry_success,
            "fallback_used": fallback_used,
        },
    }
    write_json(out_path, payload)
    print(
        f"Generated {args.output} "
        f"(count={len(rows)}, reused={reused}, updated={updated}, mock={args.mock})"
    )


if __name__ == "__main__":
    main()
