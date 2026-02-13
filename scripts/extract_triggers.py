from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from libgen import default_set_id, load_json, now_jst_iso, stable_trigger_id, write_json


RE_ASCII = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-]{1,24}")
RE_KATAKANA = re.compile(r"[ァ-ヶー]{2,16}")
RE_KANJI = re.compile(r"[一-龥々]{2,8}")
RE_QUOTES = re.compile(r"[「『【](.+?)[」』】]")


DEFAULT_FILTERS = {
    "max_doc_ratio": 0.18,
    "min_support": 3,
    "top_k": 80,
    "summary_weight": 1.0,
    "title_weight": 2.0,
    "min_token_length": 2,
    "max_token_length": 20,
    "stopwords": [],
    "exclude_exact": [],
    "exclude_suffixes": [],
    "exclude_regexes": [],
}


def normalize_token(token: str) -> str:
    t = token.strip()
    if not t:
        return ""
    if re.fullmatch(r"[A-Za-z0-9_+\-]+", t):
        t = t.lower()
    return t


def is_token_allowed(token: str, filters: dict, exclude_res: list[re.Pattern[str]]) -> bool:
    t = normalize_token(token)
    if not t:
        return False
    min_len = int(filters.get("min_token_length", 2))
    max_len = int(filters.get("max_token_length", 20))
    if not (min_len <= len(t) <= max_len):
        return False

    stopwords = set(filters.get("stopwords", []))
    if t in stopwords:
        return False
    if t in set(filters.get("exclude_exact", [])):
        return False
    for suffix in filters.get("exclude_suffixes", []):
        if suffix and t.endswith(suffix):
            return False
    for regex in exclude_res:
        if regex.search(t):
            return False
    return True


def tokenize_text(text: str, filters: dict, exclude_res: list[re.Pattern[str]]) -> set[str]:
    tokens: set[str] = set()
    if not text:
        return tokens
    normalized = text.strip().replace("\u3000", " ")

    for m in RE_QUOTES.findall(normalized):
        t = normalize_token(m)
        if is_token_allowed(t, filters, exclude_res):
            tokens.add(t)

    for m in RE_ASCII.findall(normalized):
        token = normalize_token(m)
        if is_token_allowed(token, filters, exclude_res):
            tokens.add(token)
    for m in RE_KATAKANA.findall(normalized):
        token = normalize_token(m)
        if is_token_allowed(token, filters, exclude_res):
            tokens.add(token)
    for m in RE_KANJI.findall(normalized):
        token = normalize_token(m)
        if is_token_allowed(token, filters, exclude_res):
            tokens.add(token)
    return tokens


def build_trigger_candidates(
    archives_path: Path,
    set_id: str,
    min_support: float,
    max_doc_ratio: float,
    top_k: int,
    filters: dict,
) -> dict:
    payload = load_json(archives_path)
    items = payload.get("items", [])
    docs = [
        i
        for i in items
        if (i.get("summary_ai", "").strip() or i.get("title", "").strip())
    ]
    total_docs = len(docs)
    if total_docs == 0:
        raise ValueError("No analyzable docs found in archives.json")

    exclude_res = [re.compile(p) for p in filters.get("exclude_regexes", [])]
    support: dict[str, set[str]] = defaultdict(set)
    summary_support: dict[str, set[str]] = defaultdict(set)
    title_support: dict[str, set[str]] = defaultdict(set)
    weighted_support: dict[str, float] = defaultdict(float)
    samples: dict[str, list[str]] = defaultdict(list)
    summary_weight = float(filters.get("summary_weight", 1.0))
    title_weight = float(filters.get("title_weight", 2.0))
    if summary_weight < 0 or title_weight < 0:
        raise ValueError("summary_weight/title_weight must be >= 0")
    for doc in docs:
        video_id = doc["video_id"]
        title = doc.get("title", "")
        summary_tokens = tokenize_text(doc.get("summary_ai", ""), filters, exclude_res)
        title_tokens = tokenize_text(title, filters, exclude_res)
        merged = summary_tokens | title_tokens

        for token in merged:
            support[token].add(video_id)
            if token in summary_tokens:
                summary_support[token].add(video_id)
                weighted_support[token] += summary_weight
            if token in title_tokens:
                title_support[token].add(video_id)
                weighted_support[token] += title_weight
            if len(samples[token]) < 2 and title:
                samples[token].append(title)

    filtered: list[dict] = []
    for token, vids in support.items():
        s = len(vids)
        ws = weighted_support[token]
        if ws < min_support:
            continue
        ratio = s / total_docs
        if ratio > max_doc_ratio:
            continue
        specificity = (1.0 - ratio) ** 2
        score = round(ws * specificity, 6)
        filtered.append(
            {
                "trigger": token,
                "support": s,
                "weighted_support": round(ws, 4),
                "summary_support": len(summary_support.get(token, set())),
                "title_support": len(title_support.get(token, set())),
                "doc_ratio": round(ratio, 4),
                "score": score,
                "video_ids": sorted(vids),
                "sample_titles": samples.get(token, []),
            }
        )

    filtered.sort(
        key=lambda x: (
            -x["score"],
            -x["weighted_support"],
            -x["title_support"],
            -x["support"],
            x["doc_ratio"],
            x["trigger"],
        )
    )
    filtered = filtered[:top_k]

    existing: set[str] = set()
    for item in filtered:
        item["trigger_id"] = stable_trigger_id(item["trigger"], existing)

    return {
        "set_id": set_id,
        "generated_at": now_jst_iso(),
        "source_archives": str(archives_path),
        "total_docs": total_docs,
        "total_summary_docs": total_docs,
        "min_support": min_support,
        "max_doc_ratio": max_doc_ratio,
        "filters": filters,
        "items": filtered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract trigger candidates from archives summaries and titles"
    )
    parser.add_argument("--archives", default="public/data/archives.json")
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument("--filters", default="config/trigger_filters.json")
    parser.add_argument("--min-support", type=float, default=None)
    parser.add_argument("--max-doc-ratio", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    filter_path = Path(args.filters)
    if filter_path.exists():
        filters = {**DEFAULT_FILTERS, **load_json(filter_path)}
    else:
        filters = dict(DEFAULT_FILTERS)

    min_support = args.min_support if args.min_support is not None else float(filters["min_support"])
    max_doc_ratio = (
        args.max_doc_ratio if args.max_doc_ratio is not None else float(filters["max_doc_ratio"])
    )
    top_k = args.top_k if args.top_k is not None else int(filters["top_k"])

    set_id = args.set_id
    output = args.output or "trigger_candidates.json"
    payload = build_trigger_candidates(
        archives_path=Path(args.archives),
        set_id=set_id,
        min_support=min_support,
        max_doc_ratio=max_doc_ratio,
        top_k=top_k,
        filters=filters,
    )
    write_json(Path(output), payload)
    print(
        f"Generated {output} "
        f"(docs={payload['total_docs']}, triggers={len(payload['items'])}, "
        f"min_support={min_support}, max_doc_ratio={max_doc_ratio})"
    )


if __name__ == "__main__":
    main()
