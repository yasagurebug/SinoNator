from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from libgen import default_set_id


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full JSON generation pipeline")
    parser.add_argument(
        "--tsv",
        default="specs/しののめにこ配信データベース - nicoライブリスト.tsv",
    )
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--publish", action="store_true", help="Also write final question_set and active_set")
    args = parser.parse_args()

    set_id = args.set_id
    trigger_file = "trigger_candidates.json"
    trigger_list_file = "trigger_list.txt"
    draft_file = "question_set_draft.json"
    final_file = "public/data/question_set.json"

    py = sys.executable
    run([py, "scripts/import_tsv.py", "--tsv", args.tsv, "--output", "public/data/archives.json"])
    run(
        [
            py,
            "scripts/extract_triggers.py",
            "--archives",
            "public/data/archives.json",
            "--set-id",
            set_id,
            "--output",
            trigger_file,
        ]
    )
    run(
        [
            py,
            "scripts/build_archive_tags.py",
            "--archives",
            "public/data/archives.json",
            "--rules",
            "config/type_normalization_rules.json",
            "--set-id",
            set_id,
            "--triggers",
            trigger_file,
            "--tags-output",
            "public/data/tags.json",
            "--archive-tags-output",
            "public/data/archive_tags.json",
        ]
    )
    run(
        [
            py,
            "scripts/prepare_trigger_list.py",
            "--set-id",
            set_id,
            "--candidates",
            trigger_file,
            "--output",
            trigger_list_file,
        ]
    )
    build_cmd = [
        py,
        "scripts/build_questionset.py",
        "--set-id",
        set_id,
        "--templates",
        "config/question_templates.json",
        "--triggers",
        trigger_file,
        "--trigger-list",
        trigger_list_file,
        "--draft-output",
        draft_file,
    ]
    if args.publish:
        build_cmd.extend(["--final-output", final_file, "--activate"])
    run(build_cmd)

    run([py, "scripts/validate_questionset.py", draft_file])
    if args.publish:
        run([py, "scripts/validate_questionset.py", final_file])

    print("Pipeline done.")


if __name__ == "__main__":
    main()
