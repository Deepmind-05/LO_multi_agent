import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from agent1.agent import run_agent1
from agent2.agent import run_agent2
from agent3.agent import run_agent3, run_agent3_revision
from agent4.agent import run_agent4, parse_validation

BASE_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    return BASE_DIR / path


def save_text(path: Path, content: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def load_input_records(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as err:
                    print(f"Warning: Skipping invalid JSON on line {line_idx}: {err}")
        else:
            # Supports .json (single object or list of objects)
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
    return records


def get_question_id(item: dict, index: int) -> str:
    qid = None
    if "id" in item:
        qid = str(item["id"])
    elif "question_id" in item:
        qid = str(item["question_id"])
    elif isinstance(item.get("concepts"), dict) and "question_id" in item["concepts"]:
        qid = str(item["concepts"]["question_id"])
    elif isinstance(item.get("question"), dict) and "id" in item["question"]:
        qid = str(item["question"]["id"])

    if not qid:
        qid = f"question_{index + 1}"

    sanitized = re.sub(r"[^\w\-_\.]", "_", qid).strip("_")
    return sanitized or f"question_{index + 1}"


def extract_target_languages(item: dict, default_lang: str) -> list[str]:
    raw = (
        item.get("target_languages")
        or item.get("target_language")
        or item.get("target_language_id")
        or item.get("target_languages_id")
    )
    if not raw:
        return [default_lang] if default_lang else []

    if isinstance(raw, list):
        langs = []
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                langs.append(entry.strip())
            elif isinstance(entry, dict):
                lid = entry.get("language_id") or entry.get("id")
                if lid:
                    langs.append(str(lid).strip())
        return langs or ([default_lang] if default_lang else [])

    if isinstance(raw, str):
        if "," in raw:
            return [x.strip() for x in raw.split(",") if x.strip()]
        return [raw.strip()]

    return [default_lang] if default_lang else []


def process_question_target(
    question_record: dict,
    question_id: str,
    target_language: str,
    paths: dict,
    max_retries: int,
) -> dict:
    """Runs the 4-agent pipeline for a single question and target language."""
    languages_path = resolve_path(paths["languages"])
    agent1_prompt_path = resolve_path(paths["agent1_prompt"])
    agent2_prompt_path = resolve_path(paths["agent2_prompt"])
    agent3_prompt_path = resolve_path(paths["agent3_prompt"])
    agent3_revision_prompt_path = resolve_path(paths["agent3_revision_prompt"])
    agent4_prompt_path = resolve_path(paths["agent4_prompt"])

    # Output directory per question and target language
    base_out_dir = resolve_path(paths.get("output_dir", "outputs"))
    out_dir = base_out_dir / question_id / target_language
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts_path = out_dir / "concepts.txt"
    rationale_path = out_dir / "rationale.txt"
    puzzle_path = out_dir / "puzzle.txt"
    validation_path = out_dir / "validation.txt"
    failed_puzzle_path = out_dir / "failed_puzzle.txt"
    failed_validation_path = out_dir / "failed_validation.txt"

    print(f"\n========================================================")
    print(f"Processing: [{question_id}] -> Target Language: [{target_language}]")
    print(f"========================================================")

    # --------------------------------------------------
    # AGENT 1 — Concept Architect (WHAT)
    # --------------------------------------------------
    print("Running Agent 1 (Concept Architect)...")
    agent1_result = run_agent1(
        language_id=target_language,
        source_input=question_record,
        languages_path=str(languages_path),
        prompt_path=str(agent1_prompt_path),
    )
    save_text(concepts_path, agent1_result)
    print("Agent 1 complete.")

    # --------------------------------------------------
    # AGENT 2 — Rationale Architect (HOW)
    # --------------------------------------------------
    print("Running Agent 2 (Rationale Architect)...")
    agent2_result = run_agent2(
        agent1_output_path=str(concepts_path),
        source_input=question_record,
        languages_path=str(languages_path),
        language_id=target_language,
        prompt_path=str(agent2_prompt_path),
    )
    save_text(rationale_path, agent2_result)
    print("Agent 2 complete.")

    # --------------------------------------------------
    # AGENT 3 — Problem Generator (BUILD)
    # --------------------------------------------------
    print("Running Agent 3 (Problem Generator)...")
    puzzle = run_agent3(
        source_input=question_record,
        agent1_output_path=str(concepts_path),
        agent2_output_path=str(rationale_path),
        languages_path=str(languages_path),
        language_id=target_language,
        prompt_path=str(agent3_prompt_path),
    )
    save_text(puzzle_path, puzzle)
    print("Agent 3 complete.")

    # --------------------------------------------------
    # AGENT 4 — Puzzle Validator (BREAK / ACCEPT)
    # --------------------------------------------------
    final_verdict = "FAIL"
    final_difficulty = "Not specified"
    attempts_used = 0

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1
        print(f"Running Agent 4 (attempt {attempts_used}/{max_retries + 1})...")
        validation = run_agent4(
            puzzle_path=str(puzzle_path),
            source_input=question_record,
            languages_path=str(languages_path),
            language_id=target_language,
            prompt_path=str(agent4_prompt_path),
        )
        save_text(validation_path, validation)

        verdict, actual_difficulty, feedback = parse_validation(validation)
        final_verdict = verdict
        final_difficulty = actual_difficulty

        print(f"Agent 4 verdict: {verdict}")
        print(f"Agent 4 actual difficulty: {actual_difficulty}")

        # PASS
        if verdict == "PASS":
            print(f"SUCCESS: Puzzle passed validation.")
            print(f"Saved to: {puzzle_path}")
            return {
                "question_id": question_id,
                "target_language": target_language,
                "verdict": "PASS",
                "actual_difficulty": final_difficulty,
                "attempts": attempts_used,
                "output_dir": str(out_dir),
            }

        # RETRIES EXHAUSTED
        if attempt >= max_retries:
            print("FAILED: Maximum retries reached. Puzzle failed validation.")
            save_text(failed_puzzle_path, puzzle)
            save_text(failed_validation_path, validation)
            return {
                "question_id": question_id,
                "target_language": target_language,
                "verdict": verdict,
                "actual_difficulty": final_difficulty,
                "attempts": attempts_used,
                "output_dir": str(out_dir),
            }

        # REVISION
        print("Sending feedback to Agent 3...")
        print(feedback)

        puzzle = run_agent3_revision(
            puzzle_path=str(puzzle_path),
            languages_path=str(languages_path),
            language_id=target_language,
            feedback=feedback,
            revision_prompt_path=str(agent3_revision_prompt_path),
        )
        save_text(puzzle_path, puzzle)
        print(f"Agent 3 revision {attempt + 1} complete.")

    return {
        "question_id": question_id,
        "target_language": target_language,
        "verdict": final_verdict,
        "actual_difficulty": final_difficulty,
        "attempts": attempts_used,
        "output_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Linguistic Olympiad 4-Agent Pipeline (Batch JSONL Support)"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to input .jsonl file containing questions (optional, defaults to config.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_flag",
        default=None,
        help="Path to input .jsonl file (named flag option)",
    )
    parser.add_argument(
        "--target-language", "-t",
        dest="target_language_override",
        default=None,
        help="Override target language ID for all questions in this run",
    )
    parser.add_argument(
        "--max-retries", "-r",
        dest="max_retries_override",
        type=int,
        default=None,
        help="Override max retries for validation revisions",
    )
    parser.add_argument(
        "--limit", "-n", "--rows",
        dest="limit",
        type=int,
        default=None,
        help="Number of rows/questions to process from the input file",
    )
    args = parser.parse_args()

    config = load_config()
    paths = config["paths"]

    # Determine input file path
    input_arg = args.input_file or args.input_flag or paths.get("source")
    input_path = resolve_path(input_arg)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine retries and default target language
    max_retries = (
        args.max_retries_override
        if args.max_retries_override is not None
        else config.get("max_retries", 3)
    )
    default_target_lang = (
        args.target_language_override or config.get("target_language_id", "wap")
    )

    print(f"Loading input records from: {input_path}")
    records = load_input_records(input_path)
    print(f"Total questions loaded: {len(records)}")

    row_limit = (
        args.limit
        if args.limit is not None
        else config.get("limit", None)
    )
    if row_limit is not None:
        if row_limit <= 0:
            print(f"Row limit is set to {row_limit}; nothing to process.")
            return
        records = records[:row_limit]
        print(f"Limiting processing to first {len(records)} row(s).")

    if not records:
        print("No questions found in input file.")
        return

    summary = []

    for idx, question_item in enumerate(records):
        qid = get_question_id(question_item, idx)

        # Extract target languages for this question
        if args.target_language_override:
            target_langs = [args.target_language_override]
        else:
            target_langs = extract_target_languages(question_item, default_target_lang)

        print(f"\n>>> Question {idx + 1}/{len(records)} [{qid}] with targets: {target_langs}")

        for t_lang in target_langs:
            try:
                res = process_question_target(
                    question_record=question_item,
                    question_id=qid,
                    target_language=t_lang,
                    paths=paths,
                    max_retries=max_retries,
                )
                summary.append(res)
            except Exception as e:
                print(f"Error processing question [{qid}] for target [{t_lang}]: {e}")
                summary.append({
                    "question_id": qid,
                    "target_language": t_lang,
                    "verdict": f"ERROR: {e}",
                    "actual_difficulty": "N/A",
                    "attempts": 0,
                    "output_dir": "",
                })

    # Summary Report
    print("\n" + "=" * 60)
    print("BATCH EXECUTION SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for s in summary if s.get("verdict") == "PASS")
    print(f"Total Tasks: {len(summary)} | Passed: {passed_count} | Failed: {len(summary) - passed_count}\n")

    for s in summary:
        v = s.get("verdict")
        mark = "✓" if v == "PASS" else "✗"
        print(f" {mark} [{s['question_id']}] -> Language: {s['target_language']} | Verdict: {v} | Attempts: {s.get('attempts')}")

    # Write summary json
    base_out_dir = resolve_path(paths.get("output_dir", "outputs"))
    summary_file = base_out_dir / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull summary written to: {summary_file}")


if __name__ == "__main__":
    main()
