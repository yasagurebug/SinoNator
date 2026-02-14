from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


JST = timezone(timedelta(hours=9))


def now_jst_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def default_set_id() -> str:
    return datetime.now(JST).strftime("%Y-%m")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=str(path.parent),
        delete=False,
    ) as tf:
        json.dump(payload, tf, ensure_ascii=False, indent=2)
        tf.write("\n")
        temp_name = tf.name
    Path(temp_name).replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            out.append(json.loads(t))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def stable_id(prefix: str, *parts: str) -> str:
    raw = "||".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def parse_json_string(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("LLM response does not contain JSON object")
    return json.loads(m.group(0))


def _extract_text_from_responses_api(payload: dict[str, Any]) -> str:
    out_text = payload.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text
    for out in payload.get("output", []):
        for content in out.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError("Could not extract text from OpenAI responses payload")


def _openai_http_post(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_sec: int = 120,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def call_openai_json(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: dict[str, Any],
    model: str,
    api_key: str,
    max_output_tokens: int = 800,
    temperature: float = 0.2,
    retries: int = 2,
) -> dict[str, Any]:
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError("OPENAI_API_KEY is missing. Set real key in env (placeholder not allowed).")

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }

    last_err: Exception | None = None
    for i in range(retries + 1):
        try:
            raw = _openai_http_post(
                "https://api.openai.com/v1/responses",
                api_key=api_key,
                payload=payload,
            )
            text = _extract_text_from_responses_api(raw)
            return parse_json_string(text)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
            last_err = e
            if i < retries:
                time.sleep(1.2 * (i + 1))
                continue
            break
    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")


def load_pipeline_config(path: Path | None) -> dict[str, Any]:
    default_path = Path("config/v2_pipeline.json")
    cfg_path = path or default_path
    if cfg_path.exists():
        return load_json(cfg_path)
    return {
        "model": "gpt-5.2",
        "openai_api_key_placeholder": "YOUR_OPENAI_API_KEY",
        "step_a": {"keywords_per_video": 8},
        "step_b": {"hook_min_support": 3, "semantic_concept_count": 24},
        "step_d": {
            "min_yes_rate": 0.05,
            "max_yes_rate": 0.95,
            "max_unknown_rate": 0.60,
            "redundant_jaccard_threshold": 0.82,
        },
        "step_e": {"variable_count": 40},
    }


def get_api_key(placeholder: str = "YOUR_OPENAI_API_KEY") -> str:
    return os.getenv("OPENAI_API_KEY", placeholder)


def build_progress_points(total_steps: int, max_logs: int) -> set[int]:
    if total_steps <= 0:
        return set()
    m = max(1, min(max_logs, total_steps))
    if total_steps <= m:
        return set(range(1, total_steps + 1))
    points: set[int] = set()
    for i in range(1, m + 1):
        p = math.ceil((i * total_steps) / m)
        points.add(min(total_steps, max(1, p)))
    points.add(total_steps)
    return points


def build_auto_progress_points(total_steps: int, reference_steps: int | None = None) -> set[int]:
    if total_steps <= 0:
        return set()
    if total_steps <= 30:
        budget = total_steps
    else:
        budget = max(12, int(math.log2(total_steps)) + 4)
    if reference_steps is not None and reference_steps > 0:
        budget = min(budget, reference_steps)
    points = build_progress_points(total_steps, budget)
    points.add(1)
    points.add(total_steps)
    return points
