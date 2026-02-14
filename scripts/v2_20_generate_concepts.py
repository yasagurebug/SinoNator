from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v2_common import (
    build_auto_progress_points,
    call_openai_json,
    get_api_key,
    load_json,
    load_pipeline_config,
    normalize_text,
    now_jst_iso,
    read_jsonl,
    stable_id,
    write_json,
)


CONCEPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "definition": {"type": "string"},
                    "seed_terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "definition", "seed_terms"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 64,
        }
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


HOOK_MERGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "canonical_name": {"type": "string"},
                    "concept_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 12,
                    },
                    "seed_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 24,
                    },
                },
                "required": ["canonical_name", "concept_ids", "seed_terms"],
                "additionalProperties": False,
            },
            "maxItems": 120,
        }
    },
    "required": ["groups"],
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


def source_fingerprint(rows: list[dict[str, Any]], archives: list[dict[str, Any]] | None = None) -> str:
    sig = hashlib.sha1()
    for r in sorted(rows, key=lambda x: str(x.get("video_id", ""))):
        sig.update(str(r.get("video_id", "")).encode("utf-8"))
        sig.update(str(r.get("input_hash", "")).encode("utf-8"))
        for kw in r.get("keywords", []):
            sig.update(normalize_text(str(kw.get("keyword", ""))).encode("utf-8"))
    if archives:
        for a in sorted(archives, key=lambda x: str(x.get("video_id", ""))):
            sig.update(str(a.get("video_id", "")).encode("utf-8"))
            sig.update(normalize_text(str(a.get("type_raw", ""))).encode("utf-8"))
    return sig.hexdigest()


def load_blocklist(path: Path) -> dict[str, list[dict[str, Any]]]:
    empty = {"rules": []}
    if not path.exists():
        return empty

    raw = load_json(path)
    out: dict[str, list[dict[str, Any]]] = {"rules": []}

    # 新形式専用: concepts[] に concepts.json の要素をそのまま貼る
    concept_rows: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        rows = raw.get("concepts", [])
        if isinstance(rows, list):
            concept_rows = [x for x in rows if isinstance(x, dict)]
    elif isinstance(raw, list):
        # 互換は不要だが、誤って配列直置きされた場合の救済として受ける
        concept_rows = [x for x in raw if isinstance(x, dict)]

    for row in concept_rows:
        name = normalize_text(str(row.get("name", "")))
        seeds: set[str] = set()
        for st in row.get("seed_terms", []):
            s = normalize_text(str(st))
            if s:
                seeds.add(s)
        if not name and not seeds:
            continue
        out["rules"].append({"name": name, "seed_terms": seeds})

    return out


def blocklist_fingerprint(block: dict[str, list[dict[str, Any]]]) -> str:
    sig = hashlib.sha1()
    for rule in sorted(
        block.get("rules", []),
        key=lambda x: (str(x.get("name", "")), ",".join(sorted(x.get("seed_terms", set())))),
    ):
        sig.update(str(rule.get("name", "")).encode("utf-8"))
        for v in sorted(rule.get("seed_terms", set())):
            sig.update(v.encode("utf-8"))
    return sig.hexdigest()


def should_exclude(concept: dict[str, Any], block: dict[str, list[dict[str, Any]]]) -> bool:
    name = normalize_text(str(concept.get("name", "")))
    seed_set = {normalize_text(str(x)) for x in concept.get("seed_terms", []) if str(x).strip()}

    for rule in block.get("rules", []):
        rule_name = str(rule.get("name", ""))
        rule_seeds: set[str] = set(rule.get("seed_terms", set()))
        if rule_name and rule_name == name:
            return True
        if rule_seeds and rule_seeds.issubset(seed_set):
            return True
    return False


def apply_blocklist(
    concepts: list[dict[str, Any]],
    block: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    for c in concepts:
        if should_exclude(c, block):
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


def merge_config_fingerprint(
    *,
    enabled: bool,
    model: str,
    max_groups: int,
    mode: str,
    input_top_k: int,
) -> str:
    raw = (
        f"enabled={enabled}||model={model}||max_groups={max_groups}"
        f"||mode={mode}||input_top_k={input_top_k}||v=hook_merge_v2"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def llm_merge_hook_groups(
    *,
    model: str,
    api_key: str,
    hooks: list[dict[str, Any]],
    max_groups: int,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for h in hooks:
        rows.append(
            {
                "concept_id": str(h.get("concept_id", "")),
                "name": str(h.get("name", "")),
                "seed_terms": [str(x) for x in h.get("seed_terms", []) if str(x).strip()],
                "support": int(h.get("support", 0)),
            }
        )

    if mode == "aggressive":
        system_prompt = (
            "You cluster Japanese hook terms for recommendation questions. "
            "Merge not only orthographic variants but also very close motifs with similar viewer intent. "
            "Do not merge broad unrelated topics. Prefer precision over recall. Return JSON only."
        )
    else:
        system_prompt = (
            "You merge Japanese hook terms for recommendation questions. "
            "Only merge near-identical variants (script/case/spelling variants). "
            "Do NOT merge merely related-but-different topics. Return JSON only."
        )
    user_prompt = (
        "Given hook concepts, output merge groups.\n"
        f"Max groups: {max_groups}\n"
        "Each group must contain 2+ concept_ids. concept_id overlap between groups is not allowed.\n"
        "If no merge is needed, return groups as an empty array.\n\n"
        "Hooks:\n" + json_dumps_compact(rows)
    )
    out = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="hook_merge_groups",
        schema=HOOK_MERGE_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=3000,
        temperature=0.0,
    )
    groups = out.get("groups", [])
    if not isinstance(groups, list):
        return []
    clean: list[dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        cid_list = [str(x).strip() for x in g.get("concept_ids", []) if str(x).strip()]
        if len(set(cid_list)) < 2:
            continue
        seed_terms = [str(x).strip() for x in g.get("seed_terms", []) if str(x).strip()]
        clean.append(
            {
                "canonical_name": str(g.get("canonical_name", "")).strip(),
                "concept_ids": list(dict.fromkeys(cid_list)),
                "seed_terms": list(dict.fromkeys(seed_terms)),
            }
        )
    return clean[: max(0, max_groups)]


def json_dumps_compact(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def apply_hook_merges(
    hooks: list[dict[str, Any]],
    merge_groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id: dict[str, dict[str, Any]] = {str(h.get("concept_id", "")): h for h in hooks}
    leader_updates: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()
    merged_groups = 0
    merged_concepts = 0
    skipped_groups = 0

    for g in merge_groups:
        ids = [cid for cid in g.get("concept_ids", []) if cid in by_id]
        ids = list(dict.fromkeys(ids))
        if len(ids) < 2:
            continue
        if any(cid in consumed for cid in ids):
            skipped_groups += 1
            continue

        candidates = [by_id[cid] for cid in ids]
        leader = sorted(
            candidates,
            key=lambda x: (-int(x.get("support", 0)), str(x.get("concept_id", ""))),
        )[0]
        leader_id = str(leader.get("concept_id", ""))

        merged_seed_terms: list[str] = []
        seen_terms: set[str] = set()
        for c in candidates:
            for t in c.get("seed_terms", []):
                ts = str(t).strip()
                if not ts:
                    continue
                nt = normalize_text(ts)
                if nt in seen_terms:
                    continue
                seen_terms.add(nt)
                merged_seed_terms.append(ts)
        for t in g.get("seed_terms", []):
            ts = str(t).strip()
            if not ts:
                continue
            nt = normalize_text(ts)
            if nt in seen_terms:
                continue
            seen_terms.add(nt)
            merged_seed_terms.append(ts)

        canonical_name = str(g.get("canonical_name", "")).strip() or str(leader.get("name", "")).strip()
        updated = dict(leader)
        updated["name"] = canonical_name
        updated["definition"] = f"文字列「{canonical_name}」に関連する話題を含む配信"
        updated["seed_terms"] = merged_seed_terms
        updated["support"] = max(int(c.get("support", 0)) for c in candidates)
        leader_updates[leader_id] = updated

        consumed.update(ids)
        merged_groups += 1
        merged_concepts += len(ids) - 1

    merged_hooks: list[dict[str, Any]] = []
    for h in hooks:
        cid = str(h.get("concept_id", ""))
        if cid in consumed:
            if cid in leader_updates:
                merged_hooks.append(leader_updates[cid])
            continue
        merged_hooks.append(h)

    return merged_hooks, {
        "merged_groups": merged_groups,
        "merged_concepts": merged_concepts,
        "skipped_groups": skipped_groups,
    }


def build_hook_concepts(
    keyword_rows: list[dict[str, Any]],
    *,
    min_support: int,
    top_k: int,
    progress_points: set[int] | None = None,
) -> list[dict[str, Any]]:
    support: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    total = len(keyword_rows)
    points = progress_points or set()
    for i, row in enumerate(keyword_rows, start=1):
        vid = str(row.get("video_id", ""))
        for kw in row.get("keywords", []):
            raw = str(kw.get("keyword", "")).strip()
            if not raw:
                continue
            nk = normalize_text(raw)
            support[nk].add(vid)
            display.setdefault(nk, raw)
        if i in points:
            print(f"[StepB] hook_scan {i}/{total} (unique_terms={len(support)})", flush=True)

    rows = []
    for nk, vids in support.items():
        s = len(vids)
        if s < min_support:
            continue
        name = display[nk]
        cid = stable_id("hook", name)
        rows.append(
            {
                "concept_id": cid,
                "type": "hook",
                "name": name,
                "definition": f"文字列「{name}」に関連する話題を含む配信",
                "seed_terms": [name],
                "support": s,
            }
        )
    rows.sort(key=lambda x: (-int(x["support"]), str(x["name"])))
    return rows[:top_k]


TYPE_SPLIT_RE = re.compile(r"[＋+／/,、・\s]+")
TYPE_ALIAS_BY_NORMALIZED = {
    "gaming": "ゲーム",
}


def normalize_type_label(raw: str) -> str:
    t = str(raw).strip()
    if not t:
        return ""
    nk = normalize_text(t)
    return TYPE_ALIAS_BY_NORMALIZED.get(nk, t)


def iter_type_tokens(type_raw: str) -> list[str]:
    base = str(type_raw).strip()
    if not base:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for s in [base] + TYPE_SPLIT_RE.split(base):
        t = normalize_type_label(s)
        if not t:
            continue
        nt = normalize_text(t)
        if nt in seen:
            continue
        seen.add(nt)
        tokens.append(t)
    return tokens


def build_core_type_concepts(
    archive_rows: list[dict[str, Any]],
    *,
    min_support: int,
    top_k: int,
    progress_points: set[int] | None = None,
) -> list[dict[str, Any]]:
    support: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    total = len(archive_rows)
    points = progress_points or set()

    for i, row in enumerate(archive_rows, start=1):
        vid = str(row.get("video_id", ""))
        for token in iter_type_tokens(str(row.get("type_raw", ""))):
            nt = normalize_text(token)
            support[nt].add(vid)
            display.setdefault(nt, token)
        if i in points:
            print(f"[StepB] core_scan {i}/{total} (unique_types={len(support)})", flush=True)

    rows: list[dict[str, Any]] = []
    for nt, vids in support.items():
        s = len(vids)
        if s < min_support:
            continue
        name = display[nt]
        rows.append(
            {
                "concept_id": stable_id("core", name),
                "type": "core",
                "name": name,
                "definition": f"TSVのタイプが「{name}」に該当する配信",
                "seed_terms": [name],
                "support": s,
            }
        )
    rows.sort(key=lambda x: (-int(x["support"]), str(x["name"])))
    return rows[:top_k]


def llm_generate_semantic_concepts(
    *,
    model: str,
    api_key: str,
    keyword_rows: list[dict[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    freq = Counter()
    for row in keyword_rows:
        for kw in row.get("keywords", []):
            t = str(kw.get("keyword", "")).strip()
            if t:
                freq[t] += 1
    sample_terms = [k for k, _ in freq.most_common(160)]

    system_prompt = (
        "You design semantic concepts for Japanese livestream recommendation questions. "
        "Return concise concepts with practical yes/no classification definitions."
    )
    user_prompt = (
        f"Create up to {max(8, target_count)} semantic concepts from these terms.\n"
        "Each concept must be broad enough to match multiple videos and include concrete seed terms.\n"
        "Terms:\n- " + "\n- ".join(sample_terms)
    )
    obj = call_openai_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema_name="semantic_concepts",
        schema=CONCEPT_SCHEMA,
        model=model,
        api_key=api_key,
        max_output_tokens=2200,
        temperature=0.3,
    )
    out = []
    for row in obj.get("concepts", []):
        name = str(row.get("name", "")).strip()
        definition = str(row.get("definition", "")).strip()
        seeds = [str(x).strip() for x in row.get("seed_terms", []) if str(x).strip()]
        if not name or not definition:
            continue
        out.append(
            {
                "concept_id": stable_id("sem", name, definition),
                "type": "semantic",
                "name": name,
                "definition": definition,
                "seed_terms": seeds[:8],
            }
        )
        if len(out) >= target_count:
            break
    return out


def mock_semantic_concepts(hooks: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    out = []
    for h in hooks[:target_count]:
        name = f"{h['name']}関連"
        definition = f"「{h['name']}」周辺の文脈や感情が中心の配信"
        out.append(
            {
                "concept_id": stable_id("sem", name, definition),
                "type": "semantic",
                "name": name,
                "definition": definition,
                "seed_terms": [h["name"]],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 Step B: generate hook/semantic concepts")
    parser.add_argument("--keywords", default="data/v2/keywords_raw.jsonl")
    parser.add_argument("--archives", default="data/v2/archives.json")
    parser.add_argument("--output", default="data/v2/concepts.json")
    parser.add_argument("--config", default="config/v2_pipeline.json")
    parser.add_argument("--blocklist", default="config/concept_blocklist.json")
    parser.add_argument("--set-id", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--allow-mock-overwrite",
        action="store_true",
        help="Allow overwriting existing output when --mock is used",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))
    model = str(cfg.get("model", "gpt-5.2"))
    placeholder = str(cfg.get("openai_api_key_placeholder", "YOUR_OPENAI_API_KEY"))
    step_b = cfg.get("step_b", {})
    hook_min_support_raw = step_b.get("hook_min_support", "auto")
    hook_top_k_raw = step_b.get("hook_top_k", "auto")
    core_type_enabled = bool(step_b.get("core_type_enabled", True))
    core_type_min_support_raw = step_b.get("core_type_min_support", "auto")
    core_type_top_k_raw = step_b.get("core_type_top_k", "auto")
    semantic_count_raw = step_b.get("semantic_concept_count", 24)
    hook_merge_enabled = bool(step_b.get("hook_merge_enabled", True))
    hook_merge_max_groups_raw = step_b.get("hook_merge_max_groups", 60)
    hook_merge_mode = str(step_b.get("hook_merge_mode", "aggressive")).strip().lower()
    hook_merge_input_top_k_raw = step_b.get("hook_merge_input_top_k", "auto")
    api_key = get_api_key(placeholder)

    keyword_rows = read_jsonl(Path(args.keywords))
    if not keyword_rows:
        raise ValueError(f"No keyword rows found: {args.keywords}")
    archives = load_json(Path(args.archives))
    archive_items = list(archives.get("items", []))
    if not archive_items:
        raise ValueError(f"No archives items found: {args.archives}")
    video_count = len(archive_items)

    hook_min_support = resolve_count_from_factor(
        hook_min_support_raw,
        base_count=video_count,
        default_factor=0.003,
        min_value=1,
        max_value=20,
    )
    hook_top_k = resolve_count_from_factor(
        hook_top_k_raw,
        base_count=video_count,
        default_factor=2.0,
        min_value=40,
        max_value=4000,
    )
    core_type_min_support = resolve_count_from_factor(
        core_type_min_support_raw,
        base_count=video_count,
        default_factor=0.01,
        min_value=1,
        max_value=20,
    )
    core_type_top_k = resolve_count_from_factor(
        core_type_top_k_raw,
        base_count=video_count,
        default_factor=0.2,
        min_value=4,
        max_value=120,
    )
    semantic_count = resolve_count_from_factor(
        semantic_count_raw,
        base_count=video_count,
        default_factor=0.0,
        min_value=0,
        max_value=120,
    )
    hook_merge_max_groups = resolve_count_from_factor(
        hook_merge_max_groups_raw,
        base_count=video_count,
        default_factor=0.2,
        min_value=0,
        max_value=200,
    )
    hook_merge_input_top_k = resolve_count_from_factor(
        hook_merge_input_top_k_raw,
        base_count=video_count,
        default_factor=0.77,
        min_value=20,
        max_value=1000,
    )
    print(f"[StepB] start rows={len(keyword_rows)} mode={'mock' if args.mock else 'api'}", flush=True)
    print(
        "[StepB] resolved "
        f"hook_min_support={hook_min_support} hook_top_k={hook_top_k} "
        f"core_type_min_support={core_type_min_support} core_type_top_k={core_type_top_k} "
        f"semantic_count={semantic_count} hook_merge_max_groups={hook_merge_max_groups} "
        f"hook_merge_input_top_k={hook_merge_input_top_k}",
        flush=True,
    )

    fp = source_fingerprint(keyword_rows, archive_items)
    block = load_blocklist(Path(args.blocklist))
    block_fp = blocklist_fingerprint(block)
    merge_fp = merge_config_fingerprint(
        enabled=hook_merge_enabled and (not args.mock),
        model=model,
        max_groups=hook_merge_max_groups,
        mode=hook_merge_mode,
        input_top_k=hook_merge_input_top_k,
    )
    out_path = Path(args.output)

    if args.mock and out_path.exists() and not args.allow_mock_overwrite:
        raise RuntimeError(
            f"--mock would overwrite existing file: {out_path}. "
            "Use a different --output or add --allow-mock-overwrite explicitly."
        )

    if out_path.exists() and not args.force:
        prev = load_json(out_path)
        prev_fp = str(prev.get("meta", {}).get("source_fingerprint", ""))
        prev_block_fp = str(prev.get("meta", {}).get("blocklist_fingerprint", ""))
        prev_merge_fp = str(prev.get("meta", {}).get("hook_merge_fingerprint", ""))
        if prev_fp == fp and prev_block_fp == block_fp and prev_merge_fp == merge_fp:
            print(f"Skipped {args.output} (source unchanged)")
            return

    hooks = build_hook_concepts(
        keyword_rows,
        min_support=hook_min_support,
        top_k=hook_top_k,
        progress_points=build_auto_progress_points(len(keyword_rows), reference_steps=len(keyword_rows)),
    )
    print(f"[StepB] hooks ready count={len(hooks)}", flush=True)

    cores: list[dict[str, Any]] = []
    if core_type_enabled:
        cores = build_core_type_concepts(
            archive_items,
            min_support=core_type_min_support,
            top_k=core_type_top_k,
            progress_points=build_auto_progress_points(len(archive_items), reference_steps=len(archive_items)),
        )
    print(f"[StepB] core(type) ready count={len(cores)}", flush=True)

    semantic: list[dict[str, Any]] = []
    if semantic_count > 0:
        if args.mock:
            print("[StepB] semantic generation (mock) ...", flush=True)
            semantic = mock_semantic_concepts(hooks, semantic_count)
        else:
            print("[StepB] semantic generation (api) ...", flush=True)
            semantic = llm_generate_semantic_concepts(
                model=model,
                api_key=api_key,
                keyword_rows=keyword_rows,
                target_count=semantic_count,
            )
    else:
        print("[StepB] semantic generation skipped (semantic_concept_count=0)", flush=True)
    print(f"[StepB] semantic ready count={len(semantic)}", flush=True)

    all_concepts = cores + hooks + semantic
    kept_concepts, dropped_by_blocklist = apply_blocklist(all_concepts, block)
    kept_core_rows = [c for c in kept_concepts if str(c.get("type", "")) == "core"]
    kept_hooks_rows = [c for c in kept_concepts if str(c.get("type", "")) == "hook"]
    kept_semantic_rows = [c for c in kept_concepts if str(c.get("type", "")) == "semantic"]
    kept_core = len(kept_core_rows)
    kept_hooks = len(kept_hooks_rows)
    kept_semantic = len(kept_semantic_rows)
    if dropped_by_blocklist > 0:
        print(f"[StepB] blocklist filtered count={dropped_by_blocklist}", flush=True)

    merge_stats = {"merged_groups": 0, "merged_concepts": 0, "skipped_groups": 0}
    if hook_merge_enabled and not args.mock and kept_hooks_rows:
        merge_input_rows = sorted(
            kept_hooks_rows,
            key=lambda x: (-int(x.get("support", 0)), str(x.get("concept_id", ""))),
        )[:hook_merge_input_top_k]
        print(
            f"[StepB] hook merge (api) target={len(merge_input_rows)}/{len(kept_hooks_rows)} "
            f"mode={hook_merge_mode}",
            flush=True,
        )
        merge_groups = llm_merge_hook_groups(
            model=model,
            api_key=api_key,
            hooks=merge_input_rows,
            max_groups=hook_merge_max_groups,
            mode=hook_merge_mode,
        )
        merged_hooks_rows, merge_stats = apply_hook_merges(kept_hooks_rows, merge_groups)
        kept_hooks_rows = merged_hooks_rows
        kept_hooks = len(kept_hooks_rows)
        print(
            f"[StepB] hook merge done groups={merge_stats['merged_groups']} "
            f"merged_concepts={merge_stats['merged_concepts']}",
            flush=True,
        )
    elif hook_merge_enabled and args.mock:
        print("[StepB] hook merge skipped in mock mode", flush=True)

    final_concepts = kept_core_rows + kept_hooks_rows + kept_semantic_rows

    payload = {
        "set_id": args.set_id or "",
        "generated_at": now_jst_iso(),
        "concepts": final_concepts,
        "meta": {
            "core_count": kept_core,
            "hook_count": kept_hooks,
            "semantic_count": kept_semantic,
            "core_count_raw": len(cores),
            "hook_count_raw": len(hooks),
            "semantic_count_raw": len(semantic),
            "filtered_by_blocklist": dropped_by_blocklist,
            "hook_merge_enabled": hook_merge_enabled and (not args.mock),
            "hook_merge_mode": hook_merge_mode,
            "hook_merge_input_top_k": hook_merge_input_top_k,
            "hook_merge_max_groups": hook_merge_max_groups,
            "hook_merge_fingerprint": merge_fp,
            "hook_merge_groups": merge_stats["merged_groups"],
            "hook_merge_concepts": merge_stats["merged_concepts"],
            "source_blocklist_path": args.blocklist,
            "blocklist_fingerprint": block_fp,
            "source_archives_path": args.archives,
            "source_keywords_path": args.keywords,
            "source_fingerprint": fp,
            "mock": args.mock,
        },
    }
    write_json(out_path, payload)
    print(
        f"Generated {args.output} "
        f"(concepts={len(payload['concepts'])}, core={kept_core}, hooks={kept_hooks}, semantic={kept_semantic}, "
        f"filtered={dropped_by_blocklist}, merged={merge_stats['merged_concepts']})"
    )


if __name__ == "__main__":
    main()
