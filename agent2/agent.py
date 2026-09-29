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


def call_agent2(
    system_prompt: str,
    agent1_output: str,
    source_rationale: str,
    target_language: dict,
    feedback: str = "",
) -> str:
    payload = {
        "AGENT1_OUTPUT": agent1_output,
        "SOURCE_RATIONALE": source_rationale,
        "TARGET_LANGUAGE": target_language,
    }
    if feedback:
        payload["VALIDATOR_FEEDBACK_AND_DIRECTIVE"] = feedback

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
                print(f"[Agent 2] API call error ({e}). Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise e


def run_agent2(
    agent1_output_path: str,
    source_input: dict | str,
    languages_path: str,
    language_id: str,
    prompt_path: str,
    feedback: str = "",
) -> str:
    prompt = load_prompt(prompt_path)
    agent1_output = load_text(agent1_output_path)
    source = load_source(source_input)
    target_profile = get_target_language_profile(language_id, languages_path)
    target_language = extract_target_language(target_profile)

    return call_agent2(
        system_prompt=prompt,
        agent1_output=agent1_output,
        source_rationale=source.get("rationale"),
        target_language=target_language,
        feedback=feedback,
    )
