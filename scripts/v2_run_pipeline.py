from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from v2_common import default_set_id


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_resume_cmd(
    *,
    py: str,
    set_id: str,
    tsv: str,
    mock: bool,
    publish: bool,
    from_step: str,
) -> str:
    cmd = [
        py,
        "scripts/v2_run_pipeline.py",
        "--set-id",
        set_id,
        "--tsv",
        tsv,
        "--from-step",
        from_step,
    ]
    if mock:
        cmd.append("--mock")
    if publish:
        cmd.append("--publish")
    return " ".join(shlex.quote(x) for x in cmd)


def print_audit_gate(
    *,
    step_code: str,
    py: str,
    set_id: str,
    tsv: str,
    mock: bool,
    publish: bool,
) -> None:
    if step_code == "B":
        print("[監査] StepB完了: `data/v2/concepts.json` を確認してください。", flush=True)
        print("[監査] 観点: 無意味concept（例: 対象全体を指す語）・重複concept・曖昧な定義。", flush=True)
        print(
            "[監査] 編集: 不要なconceptは `concepts` 配列から削除。"
            "必要に応じて `name` / `definition` / `seed_terms` を修正。",
            flush=True,
        )
        print(
            "[監査] 恒久除外: 次回以降も復活させたくない場合は "
            "`config/concept_blocklist.json` に除外ルールを追加。",
            flush=True,
        )
        print(
            "[監査] blocklist判定は `name` / `seed_terms` 基準です"
            "（`concept_id` は判定に使いません）。",
            flush=True,
        )
        print(
            "[監査] 注意: `concept_id` を変更するとStepCで再判定コストが増えるため、"
            "基本は維持してください。",
            flush=True,
        )
        resume = build_resume_cmd(
            py=py,
            set_id=set_id,
            tsv=tsv,
            mock=mock,
            publish=publish,
            from_step="C",
        )
        print(f"[監査] 停止後の再開コマンド: {resume}", flush=True)
        return

    if step_code == "D":
        print("[監査] StepD完了: `data/v2/selected_concepts.json` を確認してください。", flush=True)
        print("[監査] 観点: 概念の偏り・重複・分割性能の低い項目の混入。", flush=True)
        print(
            "[監査] 編集: 不要項目は `items` 配列から削除。"
            "必要なら順序（採用優先度）を調整。",
            flush=True,
        )
        resume = build_resume_cmd(
            py=py,
            set_id=set_id,
            tsv=tsv,
            mock=mock,
            publish=publish,
            from_step="E",
        )
        print(f"[監査] 停止後の再開コマンド: {resume}", flush=True)
        return

    if step_code == "E":
        print("[監査] StepE完了: `data/v2/questions_v2.json` を確認してください。", flush=True)
        print("[監査] 観点: 日本語の自然さ・YES/NOで答えやすいか・不適切表現の有無。", flush=True)
        print(
            "[監査] 編集: 各 `text` を修正。必要なら不適切項目を削除。"
            "`question_id` は重複しないよう維持。",
            flush=True,
        )
        resume = build_resume_cmd(
            py=py,
            set_id=set_id,
            tsv=tsv,
            mock=mock,
            publish=publish,
            from_step="F",
        )
        print(f"[監査] 停止後の再開コマンド: {resume}", flush=True)
        return

    if step_code == "F":
        print("[監査] StepF完了: `data/v2/runtime_v2_draft.json` を確認してください。", flush=True)
        print("[監査] 観点: `archives` / `questions` の件数整合、明らかな破綻データの有無。", flush=True)
        print(
            "[監査] 編集: 原則このファイルは手編集せず、上流（StepB〜E成果物）を直して再生成。",
            flush=True,
        )
        if publish:
            print("[監査] この実行では `public/data/runtime_v2.json` まで公開済みです。", flush=True)
            print("[監査] 次の作業: デプロイ・動作確認へ進んでください。", flush=True)
        else:
            print("[監査] まだ公開（publish）は実行していません。", flush=True)
            resume = build_resume_cmd(
                py=py,
                set_id=set_id,
                tsv=tsv,
                mock=mock,
                publish=True,
                from_step="F",
            )
            print(f"[監査] 公開コマンド: {resume}", flush=True)
        return


def wait_for_audit_confirmation(*, step_code: str) -> bool:
    prompt = (
        f"[監査] Step{step_code}: 監査完了なら Enter（または `yes`）で続行。"
        "停止する場合は `no` を入力: "
    )
    try:
        raw = input(prompt)
    except EOFError:
        print("[監査] 対話入力がないため、監査ゲートで停止します。", flush=True)
        return False
    answer = raw.strip().lower()
    if answer in {"", "yes", "y"}:
        return True
    if answer in {"no", "n"}:
        return False
    return False


def normalize_from_step(raw: str) -> str:
    s = (raw or "0").strip().lower()
    if s.startswith("step"):
        s = s[4:]
    if s in {"0", "a", "b", "c", "d", "e", "f"}:
        return s.upper() if s != "0" else "0"
    raise ValueError(f"Invalid --from-step: {raw} (use 0/A/B/C/D/E/F)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 question generation pipeline")
    parser.add_argument("--set-id", default=default_set_id())
    parser.add_argument(
        "--tsv",
        default="specs/しののめにこ配信データベース - nicoライブリスト.tsv",
        help="Input TSV for step0",
    )
    parser.add_argument("--mock", action="store_true", help="Do not call OpenAI API")
    parser.add_argument("--publish", action="store_true", help="Publish public/data/runtime_v2.json at StepF")
    parser.add_argument(
        "--from-step",
        default="0",
        help="Start step: 0/A/B/C/D/E/F (e.g. --from-step D to resume from StepD)",
    )
    args = parser.parse_args()

    py = sys.executable

    archives_path = "data/v2/archives.json"
    from_step = normalize_from_step(args.from_step)
    step_order = ["0", "A", "B", "C", "D", "E", "F"]
    start_index = step_order.index(from_step)

    step_a = [
        py,
        "scripts/v2_10_extract_keywords.py",
        "--archives",
        archives_path,
        "--output",
        "data/v2/keywords_raw.jsonl",
        "--set-id",
        args.set_id,
    ]
    if args.mock:
        step_a.append("--mock")

    step_b = [
        py,
        "scripts/v2_20_generate_concepts.py",
        "--keywords",
        "data/v2/keywords_raw.jsonl",
        "--archives",
        archives_path,
        "--output",
        "data/v2/concepts.json",
        "--set-id",
        args.set_id,
    ]
    if args.mock:
        step_b.append("--mock")

    step_c = [
        py,
        "scripts/v2_30_vote_concepts.py",
        "--archives",
        archives_path,
        "--concepts",
        "data/v2/concepts.json",
        "--output",
        "data/v2/concept_votes.jsonl",
    ]
    if args.mock:
        step_c.append("--mock")

    step_e = [
        py,
        "scripts/v2_50_generate_questions.py",
        "--selected",
        "data/v2/selected_concepts.json",
        "--output",
        "data/v2/questions_v2.json",
        "--config",
        "config/v2_pipeline.json",
    ]
    if args.mock:
        step_e.append("--mock")

    step_f = [
        py,
        "scripts/v2_60_build_questionset.py",
        "--set-id",
        args.set_id,
        "--selected",
        "data/v2/selected_concepts.json",
        "--questions",
        "data/v2/questions_v2.json",
        "--archives",
        archives_path,
        "--runtime-draft-output",
        "data/v2/runtime_v2_draft.json",
        "--runtime-output",
        "public/data/runtime_v2.json",
    ]
    if args.publish:
        step_f.append("--publish")

    steps: list[tuple[str, str, list[str]]] = [
        (
            "0",
            "Step0 import",
            [py, "scripts/v2_00_import_tsv.py", "--tsv", args.tsv, "--output", archives_path],
        ),
        ("A", "StepA keyword extraction", step_a),
        ("B", "StepB concept generation", step_b),
        ("C", "StepC concept voting", step_c),
        (
            "D",
            "StepD concept selection",
            [
                py,
                "scripts/v2_40_select_concepts.py",
                "--concepts",
                "data/v2/concepts.json",
                "--votes",
                "data/v2/concept_votes.jsonl",
                "--archives",
                archives_path,
                "--output",
                "data/v2/selected_concepts.json",
            ],
        ),
        ("E", "StepE question generation", step_e),
        ("F", "StepF build runtime_v2", step_f),
    ]

    run_steps = steps[start_index:]
    if from_step != "0":
        print(
            f"[Pipeline] resume from Step{from_step} "
            "(previous step outputs are required).",
            flush=True,
        )

    total_steps = len(run_steps)
    for i, (step_code, label, cmd) in enumerate(run_steps, start=1):
        print(f"[Pipeline] {i}/{total_steps} {label}", flush=True)
        run(cmd)
        if step_code in {"B", "D", "E", "F"}:
            print_audit_gate(
                step_code=step_code,
                py=py,
                set_id=args.set_id,
                tsv=args.tsv,
                mock=args.mock,
                publish=args.publish,
            )
            if not wait_for_audit_confirmation(step_code=step_code):
                print("[Pipeline] 監査ゲートで停止しました。", flush=True)
                return
            print("[Pipeline] 監査確認済み。処理を続行します。", flush=True)

    print("v2 pipeline done.")


if __name__ == "__main__":
    main()
