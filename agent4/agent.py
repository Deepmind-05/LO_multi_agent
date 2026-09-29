import json
import os
import re
import time
from pathlib import Path

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()
MODEL = "qwen-3.8-27b"
MAX_COMPLETION_TOKENS = 32768


client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY"),
)


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_source(source_input: dict | str | Path) -> dict:
    if isinstance(source_input, dict):
        return source_input
    with open(source_input, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_language_profile(
    language_id: str,
    languages_file: str,
) -> dict:
    with open(languages_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("language_id") == language_id:
                return record

    raise ValueError(
        f"Target language '{language_id}' not found in {languages_file}"
    )


def extract_target_language(target_profile: dict) -> dict:
    return {
        "language_id": target_profile.get("language_id"),
        "language_name": target_profile.get("language_name"),
        "family": target_profile.get("family"),
        "subfamily": target_profile.get("subfamily"),
        "genus": target_profile.get("genus"),
        "macroarea": target_profile.get("macroarea"),
    }


def extract_message_content(response) -> str:
    if not response or not response.choices:
        return ""
    msg = response.choices[0].message
    content = msg.content
    if content:
        return content
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    return reasoning or ""


def call_agent4(
    system_prompt: str,
    agent3_output: str,
    target_language: dict,
    target_difficulty: int,
) -> str:
    payload = {
        "AGENT3_OUTPUT": agent3_output,
        "TARGET_LANGUAGE": target_language,
        "TARGET_DIFFICULTY": target_difficulty,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                            indent=2,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.1,
                top_p=0.95,
                reasoning_effort="low",
            )
            return extract_message_content(response)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_sec = 2.0 * (2 ** attempt)
                print(f"[Agent 4] API call error ({e}). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise e


def run_agent4(
    puzzle_path: str,
    source_input: dict | str,
    languages_path: str,
    language_id: str,
    prompt_path: str,
) -> str:
    prompt = load_prompt(prompt_path)
    puzzle = load_text(puzzle_path)
    source = load_source(source_input)
    target_profile = get_target_language_profile(language_id, languages_path)
    target_language = extract_target_language(target_profile)

    return call_agent4(
        system_prompt=prompt,
        agent3_output=puzzle,
        target_language=target_language,
        target_difficulty=source.get("difficulty"),
    )


def parse_validation_dict(output: str) -> dict:
    """Safely extracts JSON dictionary from Agent 4 output."""
    cleaned = output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback bracket match
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


def parse_validation(output: str) -> tuple[str, str, str]:
    data = parse_validation_dict(output)
    if data:
        verdict = data.get("verdict", "FAIL")
        if verdict not in {"PASS", "PASS_WITH_ISSUES", "FAIL"}:
            verdict = "FAIL"
        actual_difficulty = str(data.get("actual_difficulty", "Not specified"))
        issues = data.get("issues", [])
        formatted_issues = []
        for it in issues:
            if isinstance(it, dict):
                sev = it.get("severity", "ISSUE").upper()
                desc = it.get("description", "")
                cat = it.get("category", "")
                cat_str = f" [{cat}]" if cat else ""
                formatted_issues.append(f"- [{sev}]{cat_str} {desc}")
            elif isinstance(it, str):
                formatted_issues.append(f"- {it}")

        feedback = "\n".join(formatted_issues) if formatted_issues else data.get("difficulty_justification", "")
        return verdict, actual_difficulty, feedback

    # Legacy text parsing fallback
    lines = [line.strip() for line in output.splitlines()]
    verdict = None
    actual_difficulty = None
    issues = []

    for i, line in enumerate(lines):
        if line == "VERDICT" and i + 1 < len(lines):
            verdict = lines[i + 1]
        elif line == "ACTUAL DIFFICULTY" and i + 1 < len(lines):
            actual_difficulty = lines[i + 1]
        elif line == "ISSUES":
            for issue in lines[i + 1:]:
                if issue.startswith(("- ", "* ")):
                    issues.append(issue[2:].strip())

    if verdict not in {"PASS", "PASS_WITH_ISSUES", "FAIL"}:
        for candidate in ["PASS_WITH_ISSUES", "PASS", "FAIL"]:
            if candidate in output:
                verdict = candidate
                break
        if not verdict:
            verdict = "FAIL"

    feedback = "\n".join(f"- {issue}" for issue in issues) if issues else output
    return verdict, actual_difficulty or "Not specified", feedback
