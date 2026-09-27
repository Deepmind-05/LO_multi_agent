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


def extract_target_language_and_wals(target_profile: dict) -> tuple[dict, list]:
    target_language = {
        "language_id": target_profile.get("language_id"),
        "language_name": target_profile.get("language_name"),
        "family": target_profile.get("family"),
        "subfamily": target_profile.get("subfamily"),
        "genus": target_profile.get("genus"),
        "macroarea": target_profile.get("macroarea"),
    }
    wals_evidence = target_profile.get("features", [])
    return target_language, wals_evidence


def call_agent1(
    system_prompt: str,
    source_concepts: dict,
    target_language: dict,
    wals_evidence: list,
) -> str:
    payload = {
        "SOURCE_CONCEPTS": source_concepts,
        "TARGET_LANGUAGE": target_language,
        "WALS_EVIDENCE": wals_evidence,
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
        temperature=0.2,
        top_p=0.95,
        reasoning_effort="low",
    )

    return response.choices[0].message.content


def run_agent1(
    language_id: str,
    source_input: dict | str,
    languages_path: str,
    prompt_path: str,
) -> str:
    prompt = load_prompt(prompt_path)
    source = load_source(source_input)
    target_profile = get_target_language_profile(language_id, languages_path)
    target_language, wals_evidence = extract_target_language_and_wals(target_profile)

    return call_agent1(
        system_prompt=prompt,
        source_concepts=source.get("concepts"),
        target_language=target_language,
        wals_evidence=wals_evidence,
    )
