import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

from agent1.agent import run_agent1, get_target_language_profile
from agent2.agent import run_agent2, call_agent2
from agent3.agent import run_agent3, call_agent3, call_agent3_revision, extract_target_language
from agent4.agent import run_agent4, call_agent4, parse_validation

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT_SEC = 300  # 5 minutes per question-target pair
COORDINATOR_MODEL = "qwen-3.8-27b"

cerebras_client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)


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

    sanitized = re.sub(r"[^\\w\\-_\\.]", "_", qid).strip("_")
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


COORDINATOR_SYSTEM_PROMPT = """You are the Lead Coordinator Agent for a 4-agent Linguistic Olympiad puzzle generation system:
- Agent 1 (Concept Architect): Extracts linguistic features & mappings (WHAT).
- Agent 2 (Rationale Architect): Designs step-by-step reasoning & sentence structure rules (HOW).
- Agent 3 (Problem Generator): Generates the complete puzzle dataset, questions, and answers.
- Agent 4 (Puzzle Validator): Solves and rigorously validates the puzzle (PASS/FAIL/PASS_WITH_ISSUES + critique).

YOUR RESPONSIBILITY:
After each agent step, you examine the pipeline state, diagnose validator feedback if present, and decide the NEXT ACTION:
1. "CALL_AGENT1": If concepts are missing, OR if Agent 4 identified that the foundational concepts extracted are weak, insufficient, or lack necessary typological depth.
2. "CALL_AGENT2": If rationale is missing, OR if Agent 4 identified fundamental rule contradictions, structural impossibility, or flawed reasoning in Agent 2's rationale.
3. "CALL_AGENT3": If the puzzle is missing, OR if Agent 4 identified puzzle-level defects (missing vocabulary, isolated typos, ambiguous translations, or trivial questions).
4. "CALL_AGENT4": If a new or revised puzzle was generated and needs validation (puzzle_needs_validation is true).
5. "FINISH_SUCCESS": If Agent 4 issued a PASS or PASS_WITH_ISSUES verdict and the puzzle is fully verified / acceptable.
6. "FINISH_FAIL": If and only if the total time budget (300 seconds) has expired without a passing puzzle.

DIAGNOSING VALIDATOR (AGENT 4) ISSUES & MULTI-TIER ROUTING:
When Agent 4 gives FAIL (or reports issues):
1. TIER 1 -> ROUTE TO AGENT 1 ("CALL_AGENT1"):
   - When to use: The linguistic concepts are weak, too simplistic, or lack the typological richness needed to build a valid Olympiad puzzle. Or Agent 4 states that the language does not exhibit the assumed linguistic phenomena.
   - Directive: Instruct Agent 1 to extract broader, richer, or corrected linguistic concepts.
   - Note: Rationale and puzzle will be completely cleared and regenerated from scratch using the fresh concepts.

2. TIER 2 -> ROUTE TO AGENT 2 ("CALL_AGENT2"):
   - When to use: Concepts are fine, but Agent 2's reasoning blueprint, morphological rule system, or sentence structure templates are contradictory, impossible, or structurally flawed.
   - Directive: Instruct Agent 2 to revise or redesign the pedagogical rationale and sentence structure rules.
   - Note: The puzzle will be completely cleared and regenerated from scratch using the fresh rationale.

3. TIER 3 -> ROUTE TO AGENT 3 ("CALL_AGENT3"):
   - When to use: Concepts and rationale are sound, but the puzzle implementation has concrete solvable bugs:
     * Missing vocabulary or unglossed words in the DATA section.
     * Typos in questions or incorrect entries in the answer key.
     * Ambiguous sentences requiring an additional contrast or clearer gloss.
     * Trivial questions that merely copy a data sentence.
   - Directive: Instruct Agent 3 with explicit, concise instructions to patch the puzzle directly.

4. VALIDATION ("CALL_AGENT4"):
   - Whenever a puzzle was just generated or revised (puzzle_needs_validation is true), you MUST choose "CALL_AGENT4"! Never call Agent 3 or finish without validating the latest puzzle.

- There is NO limit on retries; generation continues iteratively until either a passing puzzle is accepted ("FINISH_SUCCESS") or the 300-second time budget expires ("FINISH_FAIL").
- Always output valid JSON strictly conforming to the schema below.

OUTPUT JSON SCHEMA:
{
  "reasoning": "Brief explanation of your evaluation, diagnosis of issues, and rationale for which agent to call",
  "decision": "CALL_AGENT1" | "CALL_AGENT2" | "CALL_AGENT3" | "CALL_AGENT4" | "FINISH_SUCCESS" | "FINISH_FAIL",
  "target_agent": "Agent1" | "Agent2" | "Agent3" | "Agent4" | "None",
  "directive": "Specific instructions or feedback to pass to the target agent (if any)"
}
"""


def extract_coordinator_content(response) -> str:
    if not response or not response.choices:
        return ""
    msg = response.choices[0].message
    content = msg.content
    if content:
        return content
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return reasoning or ""


def call_coordinator_decision(state: dict) -> dict:
    snapshot = {
        "question_id": state["question_id"],
        "target_language": state["target_language"],
        "elapsed_seconds": round(state["elapsed_seconds"], 1),
        "time_remaining_seconds": max(0, round(state["time_limit_seconds"] - state["elapsed_seconds"], 1)),
        "has_concepts": bool(state.get("concepts")),
        "concepts_length": len(state.get("concepts", "") or ""),
        "has_rationale": bool(state.get("rationale")),
        "rationale_length": len(state.get("rationale", "") or ""),
        "has_puzzle": bool(state.get("puzzle")),
        "puzzle_length": len(state.get("puzzle", "") or ""),
        "puzzle_needs_validation": bool(state.get("puzzle") and not state.get("validation")),
        "last_verdict": state.get("last_verdict"),
        "validator_feedback": state.get("validator_feedback", "")[:1500] if state.get("validator_feedback") else None,
        "history": state.get("history", [])[-5:],
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = cerebras_client.chat.completions.create(
                model=COORDINATOR_MODEL,
                messages=[
                    {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(snapshot, indent=2)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=2048,
            )
            raw = extract_coordinator_content(response).strip()
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in raw:
                raw = raw.split("```", 1)[1].split("```", 1)[0].strip()

            if not raw:
                print("[Coordinator] Empty response from decision LLM. Using fallback.")
                return fallback_decision(state)

            decision_data = json.loads(raw)
            if not isinstance(decision_data, dict) or not decision_data.get("decision"):
                print(f"[Coordinator] Missing decision in LLM response: {raw}. Using fallback.")
                return fallback_decision(state)

            return decision_data
        except Exception as e:
            if attempt < max_retries - 1:
                wait_sec = 2.0 * (2 ** attempt)
                print(f"[Coordinator] Error calling decision LLM: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                print(f"[Coordinator] Error calling decision LLM: {e}. Using deterministic fallback.")
                return fallback_decision(state)


def fallback_decision(state: dict) -> dict:
    elapsed = state.get("elapsed_seconds", 0.0)
    time_limit = state.get("time_limit_seconds", 300.0)

    # 1. First-pass sequential checks
    if not state.get("concepts"):
        return {"decision": "CALL_AGENT1", "target_agent": "Agent1", "directive": "Generate concepts.", "reasoning": "Concepts not yet generated."}
    if not state.get("rationale"):
        return {"decision": "CALL_AGENT2", "target_agent": "Agent2", "directive": "Generate rationale.", "reasoning": "Rationale not yet generated."}
    if not state.get("puzzle"):
        return {"decision": "CALL_AGENT3", "target_agent": "Agent3", "directive": "Generate puzzle.", "reasoning": "Puzzle not yet generated."}
    
    # 2. Validation check: If puzzle was generated or revised, it MUST be validated
    if state.get("puzzle") and not state.get("validation"):
        return {"decision": "CALL_AGENT4", "target_agent": "Agent4", "directive": "Validate puzzle.", "reasoning": "Puzzle needs validation."}
    
    # 3. Acceptance check
    if state.get("last_verdict") in {"PASS", "PASS_WITH_ISSUES"}:
        return {"decision": "FINISH_SUCCESS", "target_agent": "None", "directive": "", "reasoning": f"Puzzle passed validation ({state.get('last_verdict')})."}
    
    # 4. Timeout check: ONLY fail if time budget is exhausted! No max retries!
    if elapsed >= time_limit:
        return {"decision": "FINISH_FAIL", "target_agent": "None", "directive": "", "reasoning": f"Time limit of {time_limit}s reached."}
    
    # 5. Diagnostic routing based on validator feedback
    feedback = (state.get("validator_feedback") or "").lower()

    # Tier 1: Concepts weak or missing
    if any(k in feedback for k in ["weak concept", "missing concept", "concept error", "typology not found", "wrong concept", "invalid concept", "lacks feature"]):
        return {
            "decision": "CALL_AGENT1",
            "target_agent": "Agent1",
            "directive": f"Concepts are weak or invalid according to validator: {state.get('validator_feedback', '')[:400]}",
            "reasoning": "Validator feedback indicates weak or missing concepts. Triggering Agent 1 concept re-extraction.",
        }

    # Tier 2: Rationale contradictory or structurally impossible
    if any(k in feedback for k in ["contradictory rule", "contradiction in rule", "impossible rule", "rationale contradiction", "rule conflict", "impossible grammar", "grammatical contradiction"]):
        return {
            "decision": "CALL_AGENT2",
            "target_agent": "Agent2",
            "directive": f"Rationale or rule system is contradictory according to validator: {state.get('validator_feedback', '')[:400]}",
            "reasoning": "Validator feedback indicates contradictory grammatical rules in rationale. Triggering Agent 2 revision.",
        }

    # Tier 3: Default to Agent 3 puzzle-level revision
    return {
        "decision": "CALL_AGENT3",
        "target_agent": "Agent3",
        "directive": state.get("validator_feedback", "Please revise the puzzle based on validator feedback."),
        "reasoning": "Revising puzzle implementation to address validator issues.",
    }


def run_agentic_pipeline(
    question_record: dict,
    question_id: str,
    target_language: str,
    paths: dict,
    max_retries: int = 3,
    time_limit_sec: int = DEFAULT_TIMEOUT_SEC,
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
    log_path = out_dir / "coordinator_log.json"

    target_profile = get_target_language_profile(target_language, str(languages_path))
    target_lang_dict = extract_target_language(target_profile)

    state = {
        "question_id": question_id,
        "target_language": target_language,
        "elapsed_seconds": 0.0,
        "time_limit_seconds": time_limit_sec,
        "max_retries": max_retries,
        "attempt_count": 0,
        "concepts": None,
        "rationale": None,
        "puzzle": None,
        "validation": None,
        "last_verdict": None,
        "last_difficulty": None,
        "validator_feedback": None,
        "history": [],
    }

    print(f"\n========================================================")
    print(f"[Coordinator] Starting: [{question_id}] -> Target: [{target_language}]")
    print(f"[Coordinator] Time budget: {time_limit_sec}s (strictly time-bound, continuous loop until PASS or timeout)")
    print(f"========================================================")

    step_number = 0

    while True:
        step_number += 1
        elapsed = time.monotonic() - start_time
        state["elapsed_seconds"] = elapsed

        if elapsed >= time_limit_sec:
            print(f"\n[Coordinator Watchdog] ⚠️ Time limit of {time_limit_sec}s reached for {question_id}/{target_language}.")
            state["history"].append({
                "step": step_number,
                "action": "TIMEOUT",
                "reasoning": f"Elapsed time {round(elapsed, 1)}s exceeded {time_limit_sec}s budget.",
            })
            act = "FINISH_FAIL"
            break

        print(f"\n[Coordinator Turn {step_number}] Assessing state (Elapsed: {round(elapsed, 1)}s / {time_limit_sec}s)...")
        decision = call_coordinator_decision(state)
        act = decision.get("decision")
        if not act:
            print("[Coordinator] Invalid or empty decision from LLM. Using fallback decision.")
            decision = fallback_decision(state)
            act = decision.get("decision", "FINISH_FAIL")

        reasoning = decision.get("reasoning", "")
        directive = decision.get("directive", "")
        print(f" -> Decision: {act} | Reasoning: {reasoning}")

        state["history"].append({
            "step": step_number,
            "decision": act,
            "reasoning": reasoning,
            "directive": directive,
            "elapsed_seconds": round(elapsed, 1),
        })

        if act == "FINISH_SUCCESS":
            print(f"[Coordinator] ✅ Puzzle successfully validated and accepted!")
            break

        if act == "FINISH_FAIL":
            print(f"[Coordinator] ❌ Concluding task as FAILED ({reasoning}).")
            break

        if act == "CALL_AGENT1":
            print("[Coordinator -> Agent 1] Extracting / Revising concepts...")
            concepts = run_agent1(
                language_id=target_language,
                source_input=question_record,
                languages_path=str(languages_path),
                prompt_path=str(agent1_prompt_path),
                feedback=directive,
            )
            state["concepts"] = concepts
            save_text(concepts_path, concepts)
            # TIER 1 RESET: Clear downstream rationale and puzzle so they regenerate fresh with new concepts
            state["rationale"] = None
            state["puzzle"] = None
            state["validation"] = None
            state["last_verdict"] = None
            state["validator_feedback"] = None
            if not concepts or not concepts.strip():
                print("[Coordinator] ⚠️ Agent 1 returned empty output.")
            else:
                print(f"[Coordinator] Agent 1 complete ({len(concepts)} chars). Cleared downstream rationale and puzzle.")

        elif act == "CALL_AGENT2":
            print("[Coordinator -> Agent 2] Generating / Revising rationale...")
            if directive:
                system_prompt = agent2_prompt_path.read_text(encoding="utf-8")
                combined_prompt = f"{system_prompt}\n\nCOORDINATOR DIRECTIVE / VALIDATOR FEEDBACK:\n{directive}"
                rationale = call_agent2(
                    system_prompt=combined_prompt,
                    agent1_output=state.get("concepts", "") or "",
                    source_rationale=question_record.get("rationale", "") or "",
                    target_language=target_lang_dict,
                    feedback=directive,
                )
            else:
                rationale = run_agent2(
                    agent1_output_path=str(concepts_path),
                    source_input=question_record,
                    languages_path=str(languages_path),
                    language_id=target_language,
                    prompt_path=str(agent2_prompt_path),
                )
            state["rationale"] = rationale
            save_text(rationale_path, rationale)
            # TIER 2 RESET: Clear downstream puzzle so it regenerates fresh with new rationale
            state["puzzle"] = None
            state["validation"] = None
            state["last_verdict"] = None
            state["validator_feedback"] = None
            if not rationale or not rationale.strip():
                print("[Coordinator] ⚠️ Agent 2 returned empty output.")
            else:
                print(f"[Coordinator] Agent 2 complete ({len(rationale)} chars). Cleared downstream puzzle.")

        elif act == "CALL_AGENT3":
            print("[Coordinator -> Agent 3] Generating / Revising puzzle...")
            state["attempt_count"] += 1
            if state.get("puzzle") and (directive or state.get("validator_feedback")):
                feedback_to_send = directive or state.get("validator_feedback", "")
                system_prompt = agent3_revision_prompt_path.read_text(encoding="utf-8")
                puzzle = call_agent3_revision(
                    system_prompt=system_prompt,
                    current_puzzle=state.get("puzzle", "") or "",
                    target_language=target_lang_dict,
                    feedback=feedback_to_send,
                    agent1_output=state.get("concepts", "") or "",
                    agent2_rationale=state.get("rationale", "") or "",
                    source_question=question_record.get("question"),
                )
            else:
                system_prompt = agent3_prompt_path.read_text(encoding="utf-8")
                if directive:
                    system_prompt = f"{system_prompt}\n\nSPECIAL DIRECTIVE FROM COORDINATOR:\n{directive}"
                puzzle = call_agent3(
                    system_prompt=system_prompt,
                    source_question=question_record.get("question"),
                    agent1_output=state.get("concepts", "") or "",
                    agent2_rationale=state.get("rationale", "") or "",
                    target_language=target_lang_dict,
                    target_difficulty=question_record.get("difficulty"),
                )
            state["puzzle"] = puzzle
            save_text(puzzle_path, puzzle)
            # TIER 3 RESET: Reset validation state so new/revised puzzle gets validated by Agent 4
            state["validation"] = None
            state["last_verdict"] = None
            state["validator_feedback"] = None
            if not puzzle or not puzzle.strip():
                print("[Coordinator] ⚠️ Agent 3 returned empty/truncated output. Will re-prompt.")
            else:
                print(f"[Coordinator] Agent 3 complete ({len(puzzle)} chars).")

        elif act == "CALL_AGENT4":
            print("[Coordinator -> Agent 4] Validating current puzzle...")
            validation = run_agent4(
                puzzle_path=str(puzzle_path),
                source_input=question_record,
                languages_path=str(languages_path),
                language_id=target_language,
                prompt_path=str(agent4_prompt_path),
            )
            state["validation"] = validation
            save_text(validation_path, validation)

            verdict, actual_difficulty, feedback = parse_validation(validation)
            state["last_verdict"] = verdict
            state["last_difficulty"] = actual_difficulty
            state["validator_feedback"] = feedback
            print(f"[Coordinator] Agent 4 verdict: {verdict} | Difficulty: {actual_difficulty}")

        time.sleep(1.0)

    save_text(log_path, json.dumps(state["history"], indent=2))

    is_success = (act == "FINISH_SUCCESS") or (state.get("last_verdict") in {"PASS", "PASS_WITH_ISSUES"})
    return {
        "question_id": question_id,
        "target_language": target_language,
        "verdict": state.get("last_verdict") if is_success else "FAIL",
        "actual_difficulty": state.get("last_difficulty") or "N/A",
        "attempts": state.get("attempt_count", 0),
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
        "output_dir": str(out_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent Coordinator for Linguistic Olympiad Puzzles"
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
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Timeout in seconds per question-target pair (default: 300s / 5 mins)",
    )
    args = parser.parse_args()

    config = load_config()
    paths = config["paths"]

    input_arg = args.input_file or args.input_flag or paths.get("source")
    input_path = resolve_path(input_arg)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    max_retries = (
        args.max_retries_override
        if args.max_retries_override is not None
        else config.get("max_retries", 3)
    )
    default_target_lang = (
        args.target_language_override or config.get("target_language_id", "wap")
    )
    timeout_sec = args.timeout or DEFAULT_TIMEOUT_SEC

    print(f"Loading input records from: {input_path}")
    records = load_input_records(input_path)
    print(f"Total questions loaded: {len(records)}")

    row_limit = args.limit if args.limit is not None else config.get("limit", None)
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

        if args.target_language_override:
            target_langs = [args.target_language_override]
        else:
            target_langs = extract_target_languages(question_item, default_target_lang)

        print(f"\n========================================================")
        print(f">>> Question {idx + 1}/{len(records)} [{qid}] with targets: {target_langs}")
        print(f"========================================================")

        for t_lang in target_langs:
            try:
                res = run_agentic_pipeline(
                    question_record=question_item,
                    question_id=qid,
                    target_language=t_lang,
                    paths=paths,
                    max_retries=max_retries,
                    time_limit_sec=timeout_sec,
                )
                summary.append(res)
            except Exception as e:
                print(f"[Coordinator] Error running question [{qid}] for target [{t_lang}]: {e}")
                summary.append({
                    "question_id": qid,
                    "target_language": t_lang,
                    "verdict": f"ERROR: {e}",
                    "actual_difficulty": "N/A",
                    "attempts": 0,
                    "elapsed_seconds": 0.0,
                    "output_dir": "",
                })

    print("\n" + "=" * 60)
    print("COORDINATOR BATCH EXECUTION SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for s in summary if s.get("verdict") == "PASS")
    print(f"Total Tasks: {len(summary)} | Passed: {passed_count} | Failed: {len(summary) - passed_count}\n")

    for s in summary:
        v = s.get("verdict")
        mark = "✓" if v == "PASS" else "✗"
        print(f" {mark} [{s['question_id']}] -> Language: {s['target_language']} | Verdict: {v} | Attempts: {s.get('attempts')} | Time: {s.get('elapsed_seconds')}s")

    base_out_dir = resolve_path(paths.get("output_dir", "outputs"))
    summary_file = base_out_dir / "coordinator_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFull summary written to: {summary_file}")


if __name__ == "__main__":
    main()
