import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

from agent1.agent import run_agent1, get_target_language_profile
from agent2.agent import run_agent2, call_agent2
from agent3.agent import run_agent3, call_agent3, call_agent3_revision, extract_target_language
from agent4.agent import run_agent4, parse_validation, parse_validation_dict
from db import init_db, log_puzzle_run

BASE_DIR = Path(__file__).resolve().parent
ROUTER_MODEL = "qwen-3.8-27b"

cerebras_client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)


def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def resolve_path(path: str) -> Path:
    return BASE_DIR / path


def save_text(path: Path, content: str | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def is_insufficient_data(text: str | None) -> tuple[bool, str]:
    if not text:
        return False, ""
    cleaned = text.strip()
    match = re.search(r"\bINSUFFICIENT_LANGUAGE_DATA\s*:\s*(.*)", cleaned, re.IGNORECASE)
    if match:
        reason = match.group(1).strip()
        return True, reason or "Language knowledge insufficient"
    return False, ""


def parse_range(range_str: str, total_count: int) -> list[int]:
    """
    Parses a 1-indexed range string like '1-5', '2:10', or '3' into a list of 0-indexed indices.
    """
    if not range_str:
        return list(range(total_count))

    s = range_str.strip().replace(":", "-")
    indices = []
    if "-" in s:
        parts = s.split("-", 1)
        start = max(1, int(parts[0]))
        end = min(total_count, int(parts[1]))
        indices = list(range(start - 1, end))
    else:
        single = int(s)
        if 1 <= single <= total_count:
            indices = [single - 1]
    return indices


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


def route_feedback_llm(feedback: str, question_id: str, target_lang: str) -> tuple[str, str]:
    """
    Uses Cerebras LLM to diagnose validator issues and route to Agent 1, Agent 2, or Agent 3.
    """
    system_prompt = """You are the Lead Diagnostician for a 4-agent Linguistic Olympiad puzzle system.
Given the validator's issues, determine which agent needs to fix the problem:
1. "CALL_AGENT1": If the foundational concepts extracted for the target language are completely flawed, missing, or the language lacks the assumed grammatical feature.
2. "CALL_AGENT2": If the concepts are sound, but the pedagogical rationale, grammatical rule system, or sentence structure templates are contradictory or structurally impossible.
3. "CALL_AGENT3": If concepts and rationale are valid, but the puzzle implementation has concrete bugs (e.g. missing vocabulary in DATA, typos in questions/answers, ambiguous translations, ungrammatical sentences, or difficulty adjustments).

Return valid JSON with keys:
{
  "target_agent": "Agent1" | "Agent2" | "Agent3",
  "directive": "Specific, concise guidance on what the target agent should fix."
}"""

    for _ in range(2):
        try:
            resp = cerebras_client.chat.completions.create(
                model=ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Language: {target_lang}\nFeedback:\n{feedback}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            agent = data.get("target_agent", "Agent3")
            directive = data.get("directive", feedback)
            if agent not in {"Agent1", "Agent2", "Agent3"}:
                agent = "Agent3"
            return agent, directive
        except Exception:
            pass

    # Heuristic fallback if LLM router fails
    lower = feedback.lower()
    if any(k in lower for k in ["weak concept", "missing concept", "lacks feature", "typology"]):
        return "Agent1", feedback
    if any(k in lower for k in ["contradictory rule", "impossible rule", "rationale contradiction"]):
        return "Agent2", feedback
    return "Agent3", feedback


def run_single_puzzle(
    question_record: dict,
    question_id: str,
    target_language: str,
    paths: dict,
    run_name: str,
    max_validator_attempts: int = 3,
) -> dict:
    start_time = time.monotonic()

    languages_path = resolve_path(paths["languages"])
    agent1_prompt_path = resolve_path(paths["agent1_prompt"])
    agent2_prompt_path = resolve_path(paths["agent2_prompt"])
    agent3_prompt_path = resolve_path(paths["agent3_prompt"])
    agent3_revision_prompt_path = resolve_path(paths["agent3_revision_prompt"])
    agent4_prompt_path = resolve_path(paths["agent4_prompt"])

    base_out_dir = resolve_path(paths.get("output_dir", "outputs"))
    out_dir = base_out_dir / question_id / target_language
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts_path = out_dir / "concepts.txt"
    rationale_path = out_dir / "rationale.txt"
    puzzle_path = out_dir / "puzzle.txt"
    validation_path = out_dir / "validation.txt"

    target_profile = get_target_language_profile(target_language, str(languages_path))
    target_lang_dict = extract_target_language(target_profile)
    orig_lang = question_record.get("source_language") or (
        question_record.get("concepts", {}).get("source_language")
        if isinstance(question_record.get("concepts"), dict)
        else None
    )
    target_diff = question_record.get("difficulty")

    print(f"\n========================================================")
    print(f"[{run_name}] Running: {question_id} -> Target: {target_language}")
    print(f"========================================================")

    # ----------------------------------------------------
    # Deterministic Round 1: Agent 1 -> Agent 2 -> Agent 3
    # ----------------------------------------------------
    print("--> Step 1: Agent 1 (Concept Architect)...")
    concepts = run_agent1(
        language_id=target_language,
        source_input=question_record,
        languages_path=str(languages_path),
        prompt_path=str(agent1_prompt_path),
    )
    save_text(concepts_path, concepts)
    is_insuf, reason = is_insufficient_data(concepts)
    if is_insuf:
        print(f"⚠️  Agent 1 flagged: {reason}. Skipping target language.")
        elapsed = time.monotonic() - start_time
        log_puzzle_run(
            run_name=run_name,
            question_id=question_id,
            original_language=orig_lang,
            target_language=target_language,
            target_concepts=concepts,
            target_rationale=None,
            target_puzzle=None,
            validation_data={"status": "SKIPPED_INSUFFICIENT_DATA", "reason": reason},
            verdict="SKIPPED_INSUFFICIENT_DATA",
            target_difficulty=target_diff,
            actual_difficulty=None,
            attempts_used=0,
            elapsed_seconds=elapsed,
            output_dir=str(out_dir),
        )
        return {"question_id": question_id, "target_language": target_language, "verdict": "SKIPPED_INSUFFICIENT_DATA", "attempts": 0}

    print("--> Step 2: Agent 2 (Rationale Architect)...")
    rationale = run_agent2(
        agent1_output_path=str(concepts_path),
        source_input=question_record,
        languages_path=str(languages_path),
        language_id=target_language,
        prompt_path=str(agent2_prompt_path),
    )
    save_text(rationale_path, rationale)
    is_insuf, reason = is_insufficient_data(rationale)
    if is_insuf:
        print(f"⚠️  Agent 2 flagged: {reason}. Skipping target language.")
        elapsed = time.monotonic() - start_time
        log_puzzle_run(
            run_name=run_name,
            question_id=question_id,
            original_language=orig_lang,
            target_language=target_language,
            target_concepts=concepts,
            target_rationale=rationale,
            target_puzzle=None,
            validation_data={"status": "SKIPPED_INSUFFICIENT_DATA", "reason": reason},
            verdict="SKIPPED_INSUFFICIENT_DATA",
            target_difficulty=target_diff,
            actual_difficulty=None,
            attempts_used=0,
            elapsed_seconds=elapsed,
            output_dir=str(out_dir),
        )
        return {"question_id": question_id, "target_language": target_language, "verdict": "SKIPPED_INSUFFICIENT_DATA", "attempts": 0}

    print("--> Step 3: Agent 3 (Problem Generator)...")
    puzzle = run_agent3(
        source_input=question_record,
        agent1_output_path=str(concepts_path),
        agent2_output_path=str(rationale_path),
        languages_path=str(languages_path),
        language_id=target_language,
        prompt_path=str(agent3_prompt_path),
    )
    save_text(puzzle_path, puzzle)
    is_insuf, reason = is_insufficient_data(puzzle)
    if is_insuf:
        print(f"⚠️  Agent 3 flagged: {reason}. Skipping target language.")
        elapsed = time.monotonic() - start_time
        log_puzzle_run(
            run_name=run_name,
            question_id=question_id,
            original_language=orig_lang,
            target_language=target_language,
            target_concepts=concepts,
            target_rationale=rationale,
            target_puzzle=puzzle,
            validation_data={"status": "SKIPPED_INSUFFICIENT_DATA", "reason": reason},
            verdict="SKIPPED_INSUFFICIENT_DATA",
            target_difficulty=target_diff,
            actual_difficulty=None,
            attempts_used=0,
            elapsed_seconds=elapsed,
            output_dir=str(out_dir),
        )
        return {"question_id": question_id, "target_language": target_language, "verdict": "SKIPPED_INSUFFICIENT_DATA", "attempts": 0}

    # ----------------------------------------------------
    # Validation & Reactive Revision Loop (Up to max_attempts)
    # ----------------------------------------------------
    attempts_used = 0
    final_verdict = "FAIL"
    final_act_diff = None
    last_validation_data = {}

    for attempt in range(1, max_validator_attempts + 1):
        attempts_used = attempt
        print(f"--> Step 4: Agent 4 Validation (Attempt {attempt}/{max_validator_attempts})...")
        validation = run_agent4(
            puzzle_path=str(puzzle_path),
            source_input=question_record,
            languages_path=str(languages_path),
            language_id=target_language,
            prompt_path=str(agent4_prompt_path),
        )
        save_text(validation_path, validation)
        val_dict = parse_validation_dict(validation)
        last_validation_data = val_dict or {"raw": validation}

        verdict, actual_diff_str, feedback = parse_validation(validation)
        final_verdict = verdict
        try:
            final_act_diff = int(actual_diff_str)
        except Exception:
            final_act_diff = None

        has_issues = False
        if val_dict and val_dict.get("issues"):
            has_issues = len(val_dict["issues"]) > 0
        elif feedback and feedback.strip():
            has_issues = True

        print(f"    Verdict: {verdict} | Difficulty: {actual_diff_str} | Issues present: {has_issues}")

        # Check for clean PASS: verdict must be PASS and zero issues
        if verdict == "PASS" and not has_issues:
            print(f"✅ SUCCESS: Puzzle passed validation on attempt {attempt}!")
            final_verdict = "PASS"
            break

        # If retries exhausted, conclude as FAIL
        if attempt >= max_validator_attempts:
            print(f"❌ FAILED: Maximum attempts ({max_validator_attempts}) reached without clean pass.")
            final_verdict = "FAIL"
            break

        # Otherwise, route to target agent for revision
        target_agent, directive = route_feedback_llm(feedback, question_id, target_language)
        print(f"    [Router] Directing to {target_agent}: {directive[:120]}...")

        if target_agent == "Agent1":
            print("    [Revision] Re-running Agent 1 (Concepts)...")
            concepts = run_agent1(
                language_id=target_language,
                source_input=question_record,
                languages_path=str(languages_path),
                prompt_path=str(agent1_prompt_path),
                feedback=directive,
            )
            save_text(concepts_path, concepts)
            # Downstream cascade to Agent 2 and Agent 3
            rationale = run_agent2(
                agent1_output_path=str(concepts_path),
                source_input=question_record,
                languages_path=str(languages_path),
                language_id=target_language,
                prompt_path=str(agent2_prompt_path),
            )
            save_text(rationale_path, rationale)
            puzzle = run_agent3(
                source_input=question_record,
                agent1_output_path=str(concepts_path),
                agent2_output_path=str(rationale_path),
                languages_path=str(languages_path),
                language_id=target_language,
                prompt_path=str(agent3_prompt_path),
            )
            save_text(puzzle_path, puzzle)

        elif target_agent == "Agent2":
            print("    [Revision] Re-running Agent 2 (Rationale)...")
            system_prompt = agent2_prompt_path.read_text(encoding="utf-8")
            combined_prompt = f"{system_prompt}\n\nREVISION DIRECTIVE:\n{directive}"
            rationale = call_agent2(
                system_prompt=combined_prompt,
                agent1_output=concepts,
                source_rationale=question_record.get("rationale", "") or "",
                target_language=target_lang_dict,
                feedback=directive,
            )
            save_text(rationale_path, rationale)
            # Downstream cascade to Agent 3
            puzzle = run_agent3(
                source_input=question_record,
                agent1_output_path=str(concepts_path),
                agent2_output_path=str(rationale_path),
                languages_path=str(languages_path),
                language_id=target_language,
                prompt_path=str(agent3_prompt_path),
            )
            save_text(puzzle_path, puzzle)

        else:  # Agent3
            print("    [Revision] Re-running Agent 3 (Puzzle Editor)...")
            system_prompt = agent3_revision_prompt_path.read_text(encoding="utf-8")
            puzzle = call_agent3_revision(
                system_prompt=system_prompt,
                current_puzzle=puzzle,
                target_language=target_lang_dict,
                feedback=directive,
                agent1_output=concepts,
                agent2_rationale=rationale,
                source_question=question_record.get("question"),
            )
            save_text(puzzle_path, puzzle)

    elapsed = time.monotonic() - start_time

    # Record into SQLite Database
    log_puzzle_run(
        run_name=run_name,
        question_id=question_id,
        original_language=orig_lang,
        target_language=target_language,
        target_concepts=concepts,
        target_rationale=rationale,
        target_puzzle=puzzle,
        validation_data=last_validation_data,
        verdict=final_verdict,
        target_difficulty=target_diff,
        actual_difficulty=final_act_diff,
        attempts_used=attempts_used,
        elapsed_seconds=elapsed,
        output_dir=str(out_dir),
    )

    return {
        "question_id": question_id,
        "target_language": target_language,
        "verdict": final_verdict,
        "actual_difficulty": final_act_diff,
        "attempts": attempts_used,
        "elapsed_seconds": round(elapsed, 1),
        "output_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Multilingual Linguistic Olympiad Puzzle Pipeline")
    parser.add_argument("--run-name", "-n", default=None, help="Name/tag for this experiment run (logged to DB)")
    parser.add_argument("--range", "-r", default=None, help="1-indexed row range to process e.g. '1-5', '2:10', or '3'")
    parser.add_argument("--input", "-i", default=None, help="Path to input jsonl file (defaults to config.yaml source)")
    parser.add_argument("--target-language", "-t", default=None, help="Override target language for all runs")
    parser.add_argument("--max-retries", type=int, default=3, help="Max validator passes (default 3)")

    args = parser.parse_args()

    config = load_config()
    paths = config.get("paths", {
        "source": "data/source.jsonl",
        "languages": "data/languages.jsonl",
        "agent1_prompt": "agent1/prompt.txt",
        "agent2_prompt": "agent2/prompt.txt",
        "agent3_prompt": "agent3/prompt.txt",
        "agent3_revision_prompt": "agent3/revision_prompt.txt",
        "agent4_prompt": "agent4/prompt.txt",
        "output_dir": "outputs",
    })

    input_arg = args.input or paths.get("source", "data/source.jsonl")
    input_path = resolve_path(input_arg)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    records = load_input_records(input_path)
    total_records = len(records)
    if not records:
        print("No questions found in input file.")
        return

    # Determine indices to process
    indices = parse_range(args.range, total_records)
    run_name = args.run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Initialize DB
    init_db()

    print(f"\n========================================================")
    print(f"STARTING BATCH RUN: [{run_name}]")
    print(f"Input File: {input_path} (Total records: {total_records})")
    print(f"Selected Rows: {[i + 1 for i in indices]} (Count: {len(indices)})")
    print(f"Max Validator Passes: {args.max_retries}")
    print(f"========================================================\n")

    summary = []
    default_lang = args.target_language or config.get("target_language_id", "wap")

    for idx in indices:
        question_item = records[idx]
        qid = get_question_id(question_item, idx)

        if args.target_language:
            target_langs = [args.target_language]
        else:
            target_langs = extract_target_languages(question_item, default_lang)

        for t_lang in target_langs:
            try:
                res = run_single_puzzle(
                    question_record=question_item,
                    question_id=qid,
                    target_language=t_lang,
                    paths=paths,
                    run_name=run_name,
                    max_validator_attempts=args.max_retries,
                )
                summary.append(res)
            except Exception as e:
                print(f"Error on {qid} -> {t_lang}: {e}")
                summary.append({
                    "question_id": qid,
                    "target_language": t_lang,
                    "verdict": f"ERROR: {e}",
                    "attempts": 0,
                })

    print("\n" + "=" * 60)
    print(f"SUMMARY FOR RUN: {run_name}")
    print("=" * 60)
    passed = sum(1 for s in summary if s.get("verdict") == "PASS")
    skipped = sum(1 for s in summary if s.get("verdict") == "SKIPPED_INSUFFICIENT_DATA")
    failed = len(summary) - passed - skipped
    print(f"Total: {len(summary)} | Passed: {passed} | Skipped: {skipped} | Failed: {failed}\n")
    for s in summary:
        v = s.get("verdict")
        mark = "✓" if v == "PASS" else ("⊘" if v == "SKIPPED_INSUFFICIENT_DATA" else "✗")
        print(f" {mark} [{s['question_id']}] -> {s['target_language']} | Verdict: {v} | Attempts: {s.get('attempts')}")
    print(f"\nAll results saved to DB: data/experiments.db")


if __name__ == "__main__":
    main()
