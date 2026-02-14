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
    now_jst_iso,
    normalize_text,
    read_jsonl,
    write_jsonl,
)


STEP_A_VERSION = "v6_2026_02_15_blocklist_substring_both_1"

RE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-]{1,24}|[ぁ-ゖァ-ヶー]{2,16}|[一-龥々]{2,8}")
RE_SPLIT = re.compile(r"[／/＋+・,、\s:：()（）「」『』【】\[\]<>＜＞]+")
STOPWORDS = {
    "雑談",
    "配信",
    "話題",
    "トーク",
    "内容",
    "感じ",
    "今日",
    "今回",
    "前半",
    "後半",
    "中盤",
    "終盤",
    "序盤",
}
PARTICLE_STOP = {"の", "と", "や", "で", "を", "に", "へ", "が", "は", "も", "から", "まで"}
SENTENCE_PUNCT = re.compile(r"[。．.!！?？\n]")
VERBISH_SUFFIXES = (
    "する",
    "した",
    "して",
    "される",
    "された",
    "されて",
    "できる",
    "できた",
    "ます",
    "ました",
    "です",
    "でした",
    "たい",
)


KEYWORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["keyword", "evidence"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 10,
        }
    },
    "required": ["keywords"],
    "additionalProperties": False,
}

BLOCKLIST_EXPAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 300,
        }
    },
    "required": ["terms"],
    "additionalProperties": False,
}


def input_hash(title: str, summary: str) -> str:
    raw = f"{normalize_text(title)}||{normalize_text(summary)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def is_blocked_keyword(keyword: str, blocked_norm_terms: set[str]) -> bool:
    nk = normalize_text(keyword)
    if not nk:
        return True
    for b in blocked_norm_terms:
        if not b:
            continue
        if nk == b:
            return True
        # blocked term included in keyword (e.g. "しののめにこ配信")
        if b in nk:
            return True
        # keyword is a meaningful substring of blocked term
        # (e.g. "しののめ" / "にこ" from "しののめにこ", "受肉" from "セルフ受肉")
        if len(nk) >= 2 and nk in b:
            return True
    return False


def is_keyword_word(keyword: str) -> bool:
    k = str(keyword).strip()
    if len(k) < 2 or len(k) > 24:
        return False
    if SENTENCE_PUNCT.search(k):
        return False
    if " " in k or "　" in k:
        return False
    # Must look like a compact token (proper noun / noun phrase), not a full sentence.
    if not RE_TOKEN.fullmatch(k):
        return False
    # Reject verb-like phrase endings.
    for sf in VERBISH_SUFFIXES:
        if k.endswith(sf) and len(k) >= len(sf) + 2:
            return False
    return True


def load_keyword_blocklist(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = load_json(path)
    if isinstance(payload, dict):
        rows = payload.get("terms", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    out: list[str] = []
    for x in rows:
        s = str(x).strip()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def terms_fingerprint(terms: list[str]) -> str:
    normed = sorted({normalize_text(t) for t in terms if str(t).strip()})
    raw = "||".join(normed)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def llm_expand_blocklist_terms(*, model: str, api_key: str, base_terms: list[str]) -> list[str]:
    system_prompt = (
        "You normalize Japanese blacklist terms for keyword filtering. "
        "Return only close orthographic/script/case variants and common spellings. "
        "Do not add broad new concepts."
    )
    user_prompt = (
        "Given blocked terms, output an expanded term list including notation variants.\n"
        "Keep high precision.\n"
        "Base terms:\n- " + "\n- ".join(base_terms)
    )
    obj = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="keyword_blocklist_expand",
        schema=BLOCKLIST_EXPAND_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=1200,
        temperature=0.0,
    )
    terms = [str(x).strip() for x in obj.get("terms", []) if str(x).strip()]
    return list(dict.fromkeys(terms))


def resolve_blocklist_terms(
    *,
    model: str,
    api_key: str,
    base_terms: list[str],
    cache_path: Path,
    use_llm: bool,
) -> tuple[list[str], str]:
    source_fp = terms_fingerprint(base_terms)
    if not base_terms:
        return [], source_fp

    if cache_path.exists():
        try:
            cache = load_json(cache_path)
            if (
                str(cache.get("source_fingerprint", "")) == source_fp
                and str(cache.get("model", "")) == model
                and isinstance(cache.get("terms", []), list)
            ):
                terms = [str(x).strip() for x in cache.get("terms", []) if str(x).strip()]
                return list(dict.fromkeys(terms)), source_fp
        except Exception:
            pass

    expanded = list(base_terms)
    if use_llm:
        try:
            llm_terms = llm_expand_blocklist_terms(model=model, api_key=api_key, base_terms=base_terms)
            expanded.extend(llm_terms)
        except Exception:
            # Fail-open to base_terms to avoid blocking pipeline.
            pass
    expanded = list(dict.fromkeys([t for t in expanded if str(t).strip()]))

    cache_payload = {
        "generated_at": now_jst_iso(),
        "model": model,
        "source_fingerprint": source_fp,
        "terms": expanded,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="\n") as f:
        import json

        json.dump(cache_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return expanded, source_fp


def keyword_fallback(title: str, summary: str, limit: int, blocked_norm_terms: set[str]) -> list[dict[str, str]]:
    text = f"{title}\n{summary}".strip()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in RE_TOKEN.findall(text):
        token = m.strip()
        if token in STOPWORDS:
            continue
        if not is_keyword_word(token):
            continue
        key = token.lower()
        if key in seen:
            continue
        if is_blocked_keyword(token, blocked_norm_terms):
            continue
        seen.add(key)
        found.append({"keyword": token, "evidence": token})
        if len(found) >= limit:
            break
    if not found:
        fallback = (title[:20] or summary[:20] or "その他").strip() or "その他"
        found = [{"keyword": fallback, "evidence": fallback}]
    return found


def llm_extract_keywords(
    *,
    model: str,
    api_key: str,
    title: str,
    summary: str,
    limit: int,
    blocked_terms: list[str],
) -> list[dict[str, str]]:
    blocked_for_prompt = blocked_terms[:80]
    system_prompt = (
        "You extract Japanese stream keywords as WORD tokens (single-term keywords). Return ONLY JSON. "
        "Each keyword must be present as a literal string in title or summary evidence. "
        "Avoid generic words. Do not output blocked/common terms nor their close variants. "
        "Proper nouns and compound nouns are allowed, but sentence-like phrases are forbidden."
    )
    user_prompt = (
        f"Title:\n{title}\n\n"
        f"Summary:\n{summary}\n\n"
        "Keyword constraints:\n"
        "- Output must be 単語 (one term), not a sentence.\n"
        "- OK: 固有名詞 / 複合名詞 (e.g. マッマ, GeoGuessr, 家なき子, スタバギフト敗北)\n"
        "- NG: 文・述語を含む表現 (e.g. アニメーションを描く, 〜について語る, 〜してみた)\n"
        "- Avoid terms that include explicit particle-predicate phrase forms.\n\n"
        "Blocked/common terms (and close variants must be excluded):\n- "
        + "\n- ".join(blocked_for_prompt)
        + "\n\n"
        f"Return {max(5, min(limit, 10))} keywords with short evidence snippets."
    )
    obj = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="keyword_extract",
        schema=KEYWORD_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=1200,
        temperature=0.1,
    )
    return list(obj.get("keywords", []))


def sanitize_keywords(
    rows: list[dict[str, str]],
    title: str,
    summary: str,
    primary_limit: int,
    max_limit: int,
    blocked_norm_terms: set[str],
) -> list[dict[str, str]]:
    text = f"{title}\n{summary}"
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        kw = str(row.get("keyword", "")).strip()
        ev = str(row.get("evidence", "")).strip()
        if not kw or not is_keyword_word(kw):
            continue
        if is_blocked_keyword(kw, blocked_norm_terms):
            continue
        nk = normalize_text(kw)
        if nk in seen:
            continue
        seen.add(nk)
        if ev and ev not in text and kw in text:
            ev = kw
        if ev and ev not in text:
            ev = ""
        if not ev:
            ev = kw if kw in text else (title[:24] or summary[:24] or kw)
        out.append({"keyword": kw, "evidence": ev})
        if len(out) >= primary_limit:
            break

    # Expand composite keywords into meaningful sub-terms (generalized, not term-specific).
    for row in list(out):
        if len(out) >= max_limit:
            break
        kw = str(row.get("keyword", "")).strip()
        if not kw:
            continue

        parts = [p.strip() for p in RE_SPLIT.split(kw) if p.strip()]
        candidates: list[str] = []
        for p in parts:
            if len(p) < 2:
                continue
            candidates.append(p)
            # Split by common particles in noun phrases.
            for p2 in re.split(r"(?:の|と|や|で|を|に|へ|が|は|も|から|まで)", p):
                s = p2.strip()
                if len(s) >= 2:
                    candidates.append(s)
            # Extract script-based chunks as fallback.
            candidates.extend(RE_TOKEN.findall(p))

        for c in candidates:
            cand = c.strip()
            if not is_keyword_word(cand):
                continue
            if cand in STOPWORDS or cand in PARTICLE_STOP:
                continue
            if is_blocked_keyword(cand, blocked_norm_terms):
                continue
            nc = normalize_text(cand)
            if nc in seen:
                continue
            seen.add(nc)
            ev = cand if cand in text else kw
            out.append({"keyword": cand, "evidence": ev})
            if len(out) >= max_limit:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step A: extract per-video keywords (incremental)")
    parser.add_argument("--archives", default="data/v2/archives.json")
    parser.add_argument("--output", default="data/v2/keywords_raw.jsonl")
    parser.add_argument("--config", default="config/v2_pipeline.json")
    parser.add_argument("--blocklist", default=None)
    parser.add_argument("--blocklist-cache", default=None)
    parser.add_argument("--mock", action="store_true", help="Use rule-based fallback without API call")
    parser.add_argument("--set-id", default=None)
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    model = str(cfg.get("model", "gpt-5.2"))
    placeholder = str(cfg.get("openai_api_key_placeholder", "YOUR_OPENAI_API_KEY"))
    step_a_cfg = cfg.get("step_a", {})
    primary_limit = int(step_a_cfg.get("keywords_per_video", 8))
    primary_limit = max(5, min(primary_limit, 10))
    max_limit = int(step_a_cfg.get("max_keywords_per_video", max(primary_limit, 12)))
    max_limit = max(primary_limit, min(max_limit, 30))
    api_key = get_api_key(placeholder)

    blocklist_path = Path(
        args.blocklist
        or str(step_a_cfg.get("keyword_blocklist_path", "config/keyword_blocklist.json"))
    )
    blocklist_cache_path = Path(
        args.blocklist_cache
        or str(step_a_cfg.get("keyword_blocklist_cache_path", "data/v2/keyword_blocklist_expanded.json"))
    )
    base_block_terms = load_keyword_blocklist(blocklist_path)
    expanded_block_terms, block_fp = resolve_blocklist_terms(
        model=model,
        api_key=api_key,
        base_terms=base_block_terms,
        cache_path=blocklist_cache_path,
        use_llm=not args.mock,
    )
    blocked_norm_terms = {normalize_text(x) for x in expanded_block_terms if str(x).strip()}

    archives = load_json(Path(args.archives))
    items = list(archives.get("items", []))
    if not items:
        raise ValueError("No archives items found")

    existing_rows = read_jsonl(Path(args.output))
    existing_map: dict[str, dict] = {str(r.get("video_id", "")): r for r in existing_rows}

    processed_map: dict[str, dict] = {}
    processed_count = 0
    reused = 0
    updated = 0
    total = len(items)
    progress_points = build_auto_progress_points(total, reference_steps=total)
    checkpoint_points = set(progress_points)
    checkpoint_points.add(total)
    print(
        f"[StepA] start total={total} mode={'mock' if args.mock else 'api'} "
        f"block_terms={len(base_block_terms)} expanded={len(blocked_norm_terms)}",
        flush=True,
    )

    try:
        for i, a in enumerate(items, start=1):
            video_id = str(a.get("video_id", "")).strip()
            title = str(a.get("title", "")).strip()
            summary = str(a.get("summary_ai", "")).strip()
            h = input_hash(title, summary)
            old = existing_map.get(video_id)
            action = "updated"
            if (
                old
                and old.get("input_hash") == h
                and str(old.get("extractor_version", "")) == STEP_A_VERSION
                and str(old.get("blocklist_fingerprint", "")) == block_fp
            ):
                processed_map[video_id] = old
                reused += 1
                action = "reused"
            else:
                if args.mock:
                    raw_keywords = keyword_fallback(title, summary, primary_limit, blocked_norm_terms)
                else:
                    raw_keywords = llm_extract_keywords(
                        model=model,
                        api_key=api_key,
                        title=title,
                        summary=summary,
                        limit=primary_limit,
                        blocked_terms=expanded_block_terms,
                    )
                keywords = sanitize_keywords(
                    raw_keywords,
                    title,
                    summary,
                    primary_limit=primary_limit,
                    max_limit=max_limit,
                    blocked_norm_terms=blocked_norm_terms,
                )
                if not keywords:
                    keywords = keyword_fallback(title, summary, primary_limit, blocked_norm_terms)

                processed_map[video_id] = {
                    "set_id": args.set_id or archives.get("set_id", ""),
                    "generated_at": now_jst_iso(),
                    "extractor_version": STEP_A_VERSION,
                    "blocklist_fingerprint": block_fp,
                    "video_id": video_id,
                    "title": title,
                    "summary": summary,
                    "has_summary": bool(summary),
                    "input_hash": h,
                    "keywords": keywords,
                }
                updated += 1

            processed_count += 1
            if i in checkpoint_points:
                snapshot = sorted(processed_map.values(), key=lambda r: str(r.get("video_id", "")))
                write_jsonl(Path(args.output), snapshot)

            if i in progress_points:
                print(
                    f"[StepA] {i}/{total} {video_id} {action} "
                    f"(reused={reused}, updated={updated})",
                    flush=True,
                )
    except KeyboardInterrupt:
        snapshot = sorted(processed_map.values(), key=lambda r: str(r.get("video_id", "")))
        if snapshot:
            write_jsonl(Path(args.output), snapshot)
        print(
            f"[StepA] interrupted after {processed_count}/{total} "
            f"(reused={reused}, updated={updated}, checkpoint_written={bool(snapshot)})",
            flush=True,
        )
        raise

    final_rows = sorted(processed_map.values(), key=lambda r: str(r.get("video_id", "")))
    write_jsonl(Path(args.output), final_rows)
    print(
        f"Generated {args.output} "
        f"(total={len(final_rows)}, reused={reused}, updated={updated}, mock={args.mock})"
    )


if __name__ == "__main__":
    main()
