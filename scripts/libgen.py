from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


JST = timezone(timedelta(hours=9))


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def default_set_id() -> str:
    return datetime.now(JST).strftime("%Y-%m")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_start_time(raw: str) -> str:
    value = raw.strip()
    dt = datetime.strptime(value, "%Y/%m/%d %H:%M:%S")
    return dt.replace(tzinfo=JST).isoformat(timespec="seconds")


def parse_circle_bool(raw: str) -> bool:
    return raw.strip() == "〇"


def split_type_raw(type_raw: str, delimiter: str) -> list[str]:
    text = (type_raw or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(delimiter)]
    return [p for p in parts if p]


def read_source_tsv(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if len(rows) < 3:
        raise ValueError("TSV rows are too short")

    header = rows[1]
    data_start = 3 if rows[2] == header else 2

    items: list[dict[str, str]] = []
    for row in rows[data_start:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        data = {header[i]: padded[i].strip() for i in range(len(header))}
        items.append(data)
    return items


def stable_trigger_id(trigger: str, existing: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", trigger).strip("_").lower()
    if len(base) < 3:
        base = hashlib.md5(trigger.encode("utf-8")).hexdigest()[:10]
    candidate = f"motif_{base}"
    if candidate not in existing:
        existing.add(candidate)
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    final = f"{candidate}_{suffix}"
    existing.add(final)
    return final
