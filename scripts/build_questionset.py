from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from libgen import default_set_id, load_json, now_jst_iso, write_json


def build_filter_answers() -> dict:
    return {
        "yes": {"action": "filter_in"},
        "no": {"action": "filter_out"},
    }


def normalize_key(text: str) -> str:
    t = text.strip()
    if re.fullmatch(r"[A-Za-z0-9_+\-]+", t):
        return t.lower()
    return t


def load_trigger_list(path: Path) -> tuple[list[dict], dict]:
    if not path.exists():
        return [], {"list_used": False, "active": 0, "commented": 0, "total_lines": 0}

    active: list[dict] = []
    commented = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            total += 1
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                commented += 1
                continue
            left = line
            display = ""
            if "=>" in line:
                src, disp = line.split("=>", 1)
                left = src.strip()
                display = disp.strip()

            source = left
            trigger_id = ""
            if "|" in left:
                src, tid = left.split("|", 1)
                source = src.strip()
                trigger_id = tid.strip()

            active.append(
                {
                    "source": source,
                    "display": display or source,
                    "trigger_id": trigger_id,
                }
            )
    return active, {"list_used": True, "active": len(active), "commented": commented, "total_lines": total}


def apply_trigger_list(triggers: dict, active_entries: list[dict]) -> tuple[list[dict], dict]:
    items = list(triggers.get("items", []))
    if not active_entries:
        return items, {"total_candidates": len(items), "selected": len(items), "missing_in_candidates": 0}

    index_exact = {t["trigger"]: t for t in items}
    index_norm = {normalize_key(t["trigger"]): t for t in items}
    index_id = {t["trigger_id"]: t for t in items if t.get("trigger_id")}
    selected: list[dict] = []
    seen_ids: set[str] = set()
    missing = 0
    for e in active_entries:
        source = e["source"].strip()
        display = e["display"].strip()
        entry_id = e.get("trigger_id", "").strip()
        t = index_id.get(entry_id) if entry_id else None
        if not t:
            t = index_exact.get(source)
        if not t:
            t = index_norm.get(normalize_key(source))
        if not t:
            t = index_exact.get(display)
        if not t:
            t = index_norm.get(normalize_key(display))
        if t:
            tid = t["trigger_id"]
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            row = dict(t)
            row["_display_trigger"] = display or t["trigger"]
            selected.append(row)
        else:
            missing += 1
    return selected, {
        "total_candidates": len(items),
        "selected": len(selected),
        "missing_in_candidates": missing,
    }


def validate_question_set(payload: dict) -> None:
    required_top = ["set_id", "name", "tone", "limits", "selection", "questions"]
    for key in required_top:
        if key not in payload:
            raise ValueError(f"question_set missing required key: {key}")

    questions = payload["questions"]
    if not isinstance(questions, list) or len(questions) == 0:
        raise ValueError("questions must be a non-empty array")

    ids: set[str] = set()
    for q in questions:
        for key in ["question_id", "kind", "is_active", "question_mode", "text"]:
            if key not in q:
                raise ValueError(f"question missing key={key}: {q}")
        qid = q["question_id"]
        if qid in ids:
            raise ValueError(f"duplicate question_id: {qid}")
        ids.add(qid)

        mode = q["question_mode"]
        if mode == "filter":
            if "target_tags" not in q or not q["target_tags"]:
                raise ValueError(f"filter question missing target_tags: {qid}")
            ans = q.get("answers", {})
            if ans.get("yes", {}).get("action") != "filter_in":
                raise ValueError(f"filter question yes.action invalid: {qid}")
            if ans.get("no", {}).get("action") != "filter_out":
                raise ValueError(f"filter question no.action invalid: {qid}")
        elif mode == "score":
            effects = q.get("effects", {})
            yes = effects.get("yes", [])
            no = effects.get("no", [])
            if not yes or not no:
                raise ValueError(f"score question effects missing: {qid}")
            if len(yes) != len(no):
                raise ValueError(f"score question yes/no length mismatch: {qid}")
            for y, n in zip(yes, no):
                if y["tag_id"] != n["tag_id"]:
                    raise ValueError(f"score question tag mismatch: {qid}")
                if round(float(y["weight"]) + float(n["weight"]), 6) != 0:
                    raise ValueError(f"score question weights must be opposite: {qid}")
        else:
            raise ValueError(f"invalid question_mode: {mode}")


def build_question_set(
    set_id: str,
    templates: dict,
    triggers: dict,
    trigger_list_path: Path | None = None,
) -> dict:
    randomizer = random.Random(set_id)

    fixed_questions: list[dict] = []
    for fq in templates.get("fixed_questions", []):
        q = {
            "question_id": fq["question_id"],
            "kind": "fixed",
            "is_active": bool(fq.get("is_active", True)),
            "question_mode": fq["question_mode"],
            "text": fq["text"],
        }
        if q["question_mode"] == "filter":
            q["target_tags"] = fq["target_tags"]
            q["answers"] = build_filter_answers()
        else:
            q["effects"] = fq["effects"]
        fixed_questions.append(q)

    variable_count = int(templates.get("variable_count", 10))
    variable_templates = templates.get("variable_templates", ["{trigger}の話、気になる？"])
    active_entries, list_stats = (
        load_trigger_list(trigger_list_path) if trigger_list_path else ([], {"list_used": False})
    )
    trigger_items, trigger_stats = apply_trigger_list(triggers, active_entries)
    trigger_items = trigger_items[:variable_count]
    variable_questions: list[dict] = []
    for i, t in enumerate(trigger_items):
        text_template = variable_templates[i % len(variable_templates)]
        text = text_template.format(trigger=t.get("_display_trigger", t["trigger"]))
        variable_questions.append(
            {
                "question_id": f"q_var_{t['trigger_id']}",
                "kind": "variable",
                "is_active": True,
                "question_mode": "filter",
                "text": text,
                "target_tags": [t["trigger_id"]],
                "answers": build_filter_answers(),
            }
        )
    randomizer.shuffle(variable_questions)

    payload = {
        "set_id": set_id,
        "name": f"{set_id} question set",
        "generated_at": now_jst_iso(),
        "tone": templates["tone"],
        "limits": templates["limits"],
        "selection": templates["selection"],
        "questions": fixed_questions + variable_questions,
        "meta": {
            "fixed_count": len(fixed_questions),
            "variable_count": len(variable_questions),
            "trigger_source_set_id": triggers.get("set_id"),
            "trigger_list": {
                "path": str(trigger_list_path) if trigger_list_path else "",
                **list_stats,
                **trigger_stats,
            },
        },
    }
    validate_question_set(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build question_set_draft.json")
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--templates", default="config/question_templates.json")
    parser.add_argument("--triggers", default=None, help="trigger_candidates.json path")
    parser.add_argument("--trigger-list", default=None, help="trigger_list.txt path")
    parser.add_argument("--draft-output", default=None)
    parser.add_argument("--final-output", default=None, help="Optional output for public/data/question_set.json")
    parser.add_argument("--activate", action="store_true", help="Update public/data/active_set.json when final-output is provided")
    args = parser.parse_args()

    set_id = args.set_id
    templates = load_json(Path(args.templates))

    triggers_path = args.triggers or "trigger_candidates.json"
    triggers = load_json(Path(triggers_path))
    trigger_list_path = Path(args.trigger_list) if args.trigger_list else Path("trigger_list.txt")
    payload = build_question_set(
        set_id=set_id,
        templates=templates,
        triggers=triggers,
        trigger_list_path=trigger_list_path,
    )

    draft_output = args.draft_output or "question_set_draft.json"
    write_json(Path(draft_output), payload)
    print(f"Generated {draft_output} (questions={len(payload['questions'])})")

    if args.final_output:
        write_json(Path(args.final_output), payload)
        print(f"Generated {args.final_output}")
        if args.activate:
            active = {"set_id": set_id, "updated_at": now_jst_iso()}
            write_json(Path("public/data/active_set.json"), active)
            print("Updated public/data/active_set.json")


if __name__ == "__main__":
    main()
