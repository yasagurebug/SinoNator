from __future__ import annotations

import argparse
from pathlib import Path

from libgen import default_set_id, load_json


HEADER = [
    "# Trigger word list",
    "# - One trigger per line (recommended: source|trigger_id)",
    "# - Lines starting with '#' are ignored",
    "# - To disable a trigger, prefix it with '#'",
    "# - Optional: source=>display (example: vtuber=>VTuber)",
    "# - Optional: source|trigger_id=>display (stable mapping)",
    "",
]


def parse_source_token(raw_line: str) -> str:
    line = raw_line.strip()
    if not line:
        return ""
    if line.startswith("#"):
        line = line[1:].strip()
    if not line:
        return ""
    left = line.split("=>", 1)[0].strip()
    source = left.split("|", 1)[0].strip()
    return source


def read_existing(path: Path) -> tuple[list[str], set[str]]:
    if not path.exists():
        return [], set()
    lines = path.read_text(encoding="utf-8").splitlines()
    normalized: set[str] = set()
    for line in lines:
        token = parse_source_token(line)
        if token:
            normalized.add(token)
    return lines, normalized


def build_lines(existing_lines: list[str], existing_tokens: set[str], triggers: list[dict]) -> list[str]:
    if not existing_lines:
        lines = list(HEADER)
    else:
        lines = list(existing_lines)

    to_add = [t for t in triggers if t["trigger"] not in existing_tokens]
    if not to_add:
        return lines

    if lines and lines[-1].strip():
        lines.append("")
    lines.append("# auto-added")
    for t in to_add:
        lines.append(f"{t['trigger']}|{t['trigger_id']}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create/update trigger list text file. Comment out lines with '#' to disable."
    )
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_id = args.set_id
    candidates_path = Path(args.candidates or "trigger_candidates.json")
    output_path = Path(args.output or "trigger_list.txt")

    candidates = load_json(candidates_path)
    triggers = [
        {"trigger": item["trigger"], "trigger_id": item.get("trigger_id", "")}
        for item in candidates.get("items", [])
    ]
    existing_lines, existing_tokens = read_existing(output_path)
    lines = build_lines(existing_lines, existing_tokens, triggers)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    active = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
    commented = sum(1 for l in lines if l.strip().startswith("#"))
    print(
        f"Generated {output_path} "
        f"(lines={len(lines)}, active={active}, commented={commented})"
    )


if __name__ == "__main__":
    main()
