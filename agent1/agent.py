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


def load_source_problem(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
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


def build_source_input(source: dict) -> dict:
    return {
        "source_language": source.get("language"),
        "source_rationale": source.get("rationale"),
        "source_concepts": source.get("concepts"),
        "difficulty": source.get("difficulty"),
    }

def call_agent1(
    system_prompt: str,
    source_input: dict,
    target_profile: dict,
) -> str:

    payload = {
        "SOURCE": source_input,
        "TARGET_LANGUAGE_PROFILE": target_profile,
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
    source_path: str,
    languages_path: str,
    prompt_path: str,
) -> str:

    prompt = load_prompt(prompt_path)

    source = load_source_problem(source_path)

    target_profile = get_target_language_profile(
        language_id=language_id,
        languages_file=languages_path,
    )

    source_input = build_source_input(source)

    return call_agent1(
        system_prompt=prompt,
        source_input=source_input,
        target_profile=target_profile,
    )