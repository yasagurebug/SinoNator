from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from v2_common import (
    build_auto_progress_points,
    default_set_id,
    load_json,
    now_jst_iso,
    write_json,
    write_json_atomic,
)


def validate_runtime(payload: dict[str, Any]) -> None:
    required_top = ["set_id", "name", "generated_at", "archives", "questions"]
    for key in required_top:
        if key not in payload:
            raise ValueError(f"runtime_v2 missing key: {key}")

    archives = list(payload.get("archives", []))
    questions = list(payload.get("questions", []))
    if not archives:
        raise ValueError("runtime_v2 archives is empty")
    if not questions:
        raise ValueError("runtime_v2 questions is empty")

    archive_ids = {str(a.get("video_id", "")) for a in archives}
    if "" in archive_ids:
        raise ValueError("runtime_v2 archives contain empty video_id")

    seen_qid: set[str] = set()
    for q in questions:
        for key in [
            "question_id",
            "kind",
            "question_type",
            "is_active",
            "text",
            "yes_video_ids",
            "yes_count",
            "no_count",
        ]:
            if key not in q:
                raise ValueError(f"runtime_v2 question missing {key}: {q}")

        qid = str(q["question_id"])
        if qid in seen_qid:
            raise ValueError(f"duplicate question_id: {qid}")
        seen_qid.add(qid)

        yes_ids = {str(x) for x in q.get("yes_video_ids", [])}
        unknown_ids = yes_ids - archive_ids
        if unknown_ids:
            raise ValueError(f"question has unknown yes_video_ids: {qid} / {sorted(unknown_ids)[:3]}")

        if int(q.get("yes_count", -1)) != len(yes_ids):
            raise ValueError(f"yes_count mismatch: {qid}")
        if int(q.get("no_count", -1)) != (len(archive_ids) - len(yes_ids)):
            raise ValueError(f"no_count mismatch: {qid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step F: build runtime_v2.json from v2 outputs")
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--selected", default="data/v2/selected_concepts.json")
    parser.add_argument("--questions", default="data/v2/questions_v2.json")
    parser.add_argument("--archives", default="data/v2/archives.json")
    parser.add_argument("--runtime-draft-output", default="data/v2/runtime_v2_draft.json")
    parser.add_argument("--runtime-output", default="public/data/runtime_v2.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    selected_path = Path(args.selected)
    questions_path = Path(args.questions)
    archives_path = Path(args.archives)
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{selected_path} not found. Run Step D first (scripts/v2_40_select_concepts.py)."
        )
    if not questions_path.exists():
        raise FileNotFoundError(
            f"{questions_path} not found. Run Step E first (scripts/v2_50_generate_questions.py)."
        )
    if not archives_path.exists():
        raise FileNotFoundError(f"{archives_path} not found. Run Step 0 first.")

    selected = load_json(selected_path)
    questions_payload = load_json(questions_path)
    archives_payload = load_json(archives_path)
    archives = list(archives_payload.get("items", []))
    if not archives:
        raise ValueError("archives is empty")

    selected_map = {str(x.get("concept_id", "")): x for x in selected.get("items", [])}
    question_rows = sorted(
        list(questions_payload.get("items", [])),
        key=lambda x: int(x.get("priority", 999999)),
    )

    print(
        f"[StepF] loaded archives={len(archives)} selected={len(selected_map)} questions={len(question_rows)}",
        flush=True,
    )

    total_archives = len(archives)

    variable_questions: list[dict[str, Any]] = []
    points = build_auto_progress_points(len(question_rows), reference_steps=len(archives))
    for i, q in enumerate(question_rows, start=1):
        cid = str(q.get("concept_id", "")).strip()
        sel = selected_map.get(cid)
        if not sel:
            if i in points:
                print(f"[StepF] variable_scan {i}/{len(question_rows)} (kept={len(variable_questions)})", flush=True)
            continue

        qid = str(q.get("question_id", f"qv2_{cid}")).strip()
        text = str(q.get("text", "")).strip()
        yes_ids = sorted({str(x) for x in sel.get("yes_video_ids", []) if str(x)})
        q_type = str(q.get("question_type", sel.get("type", ""))).strip().lower()
        if q_type not in {"core", "hook", "semantic"}:
            q_type = str(sel.get("type", "")).strip().lower()
        if q_type not in {"core", "hook", "semantic"}:
            if cid.startswith("core_"):
                q_type = "core"
            elif cid.startswith("hook_"):
                q_type = "hook"
            else:
                q_type = "semantic"
        if not qid or not text:
            if i in points:
                print(f"[StepF] variable_scan {i}/{len(question_rows)} (kept={len(variable_questions)})", flush=True)
            continue

        variable_questions.append(
            {
                "question_id": qid,
                "kind": "variable",
                "question_type": q_type,
                "is_active": True,
                "text": text,
                "concept_id": cid,
                "source": "v2_selected_concept",
                "yes_video_ids": yes_ids,
                "yes_count": len(yes_ids),
                "no_count": total_archives - len(yes_ids),
                "split_score": float(sel.get("split_score", 0.0)),
            }
        )
        if i in points:
            print(f"[StepF] variable_scan {i}/{len(question_rows)} (kept={len(variable_questions)})", flush=True)

    payload = {
        "set_id": args.set_id,
        "name": f"{args.set_id} runtime v2",
        "generated_at": now_jst_iso(),
        "archives": archives,
        "questions": variable_questions,
        "meta": {
            "version": "runtime_v2",
            "fixed_count": 0,
            "variable_count": len(variable_questions),
            "source_selected_path": args.selected,
            "source_questions_path": args.questions,
            "source_archives_path": args.archives,
        },
    }

    validate_runtime(payload)
    write_json(Path(args.runtime_draft_output), payload)
    print(f"Generated {args.runtime_draft_output} (questions={len(payload['questions'])})")

    if args.publish:
        write_json_atomic(Path(args.runtime_output), payload)
        print(f"Published {args.runtime_output}")


if __name__ == "__main__":
    main()
