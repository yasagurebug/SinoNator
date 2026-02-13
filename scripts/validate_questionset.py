from __future__ import annotations

import argparse
from pathlib import Path

from build_questionset import validate_question_set
from libgen import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate question_set JSON schema for MVP rules")
    parser.add_argument("path", help="Path to question_set JSON")
    args = parser.parse_args()

    payload = load_json(Path(args.path))
    validate_question_set(payload)
    print(f"OK: {args.path}")


if __name__ == "__main__":
    main()
