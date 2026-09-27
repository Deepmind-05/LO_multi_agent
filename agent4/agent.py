import json
import os
from pathlib import Path

from cerebras.cloud.sdk import Cerebras


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
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=0.1,
        top_p=0.95,
        reasoning_effort="low",
    )

    return response.choices[0].message.content


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


def parse_validation(output: str) -> tuple[str, str, str]:
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
                if issue.startswith("- "):
                    issues.append(issue[2:].strip())

    if verdict not in {"PASS", "PASS_WITH_ISSUES", "FAIL"}:
        for candidate in ["PASS", "PASS_WITH_ISSUES", "FAIL"]:
            if candidate in output:
                verdict = candidate
                break
        if not verdict:
            verdict = "FAIL"

    feedback = "\n".join(f"- {issue}" for issue in issues) if issues else output
    return verdict, actual_difficulty or "Not specified", feedback
