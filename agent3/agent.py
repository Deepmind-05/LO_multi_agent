import json
import os
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


def call_agent3(
    system_prompt: str,
    source_question: dict,
    agent1_output: str,
    agent2_rationale: str,
    target_language: dict,
    target_difficulty: int,
) -> str:
    payload = {
        "SOURCE_QUESTION": source_question,
        "AGENT1_OUTPUT": agent1_output,
        "AGENT2_RATIONALE": agent2_rationale,
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
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.2,
                top_p=0.95,
                reasoning_effort="low",
            )
            return extract_message_content(response)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_sec = 2.0 * (2 ** attempt)
                print(f"[Agent 3] API call error ({e}). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise e


def run_agent3(
    source_input: dict | str,
    agent1_output_path: str,
    agent2_output_path: str,
    languages_path: str,
    language_id: str,
    prompt_path: str,
) -> str:
    prompt = load_prompt(prompt_path)
    source = load_source(source_input)
    agent1_output = load_text(agent1_output_path)
    agent2_output = load_text(agent2_output_path)
    target_profile = get_target_language_profile(language_id, languages_path)
    target_language = extract_target_language(target_profile)

    return call_agent3(
        system_prompt=prompt,
        source_question=source.get("question"),
        agent1_output=agent1_output,
        agent2_rationale=agent2_output,
        target_language=target_language,
        target_difficulty=source.get("difficulty"),
    )


def call_agent3_revision(
    system_prompt: str,
    current_puzzle: str,
    target_language: dict,
    feedback: str,
    agent1_output: str = "",
    agent2_rationale: str = "",
    source_question: dict | None = None,
) -> str:
    if len(current_puzzle) > 20000:
        current_puzzle = current_puzzle[:20000]

    payload = {
        "CURRENT_PUZZLE": current_puzzle,
        "TARGET_LANGUAGE": target_language,
        "VALIDATOR_FEEDBACK": feedback,
    }
    if agent1_output:
        payload["AGENT1_OUTPUT"] = agent1_output
    if agent2_rationale:
        payload["AGENT2_RATIONALE"] = agent2_rationale
    if source_question:
        payload["SOURCE_QUESTION"] = source_question

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
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.2,
                top_p=0.95,
                reasoning_effort="low",
            )
            return extract_message_content(response)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_sec = 2.0 * (2 ** attempt)
                print(f"[Agent 3 Revision] API call error ({e}). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise e


def run_agent3_revision(
    puzzle_path: str,
    languages_path: str,
    language_id: str,
    feedback: str,
    revision_prompt_path: str,
    agent1_output_path: str | None = None,
    agent2_output_path: str | None = None,
    source_input: dict | str | None = None,
) -> str:
    revision_prompt = load_prompt(revision_prompt_path)
    current_puzzle = load_text(puzzle_path)
    target_profile = get_target_language_profile(language_id, languages_path)
    target_language = extract_target_language(target_profile)

    agent1_output = load_text(agent1_output_path) if agent1_output_path and Path(agent1_output_path).exists() else ""
    agent2_output = load_text(agent2_output_path) if agent2_output_path and Path(agent2_output_path).exists() else ""
    source = load_source(source_input) if source_input else {}

    return call_agent3_revision(
        system_prompt=revision_prompt,
        current_puzzle=current_puzzle,
        target_language=target_language,
        feedback=feedback,
        agent1_output=agent1_output,
        agent2_rationale=agent2_output,
        source_question=source.get("question") if source else None,
    )
