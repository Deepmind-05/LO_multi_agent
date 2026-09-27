import os
from pathlib import Path
import json

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


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def call_agent2(
    system_prompt: str,
    source_question: dict,
    agent1_output: str,
) -> str:

    payload = {
        "SOURCE_QUESTION": source_question,
        "AGENT1_OUTPUT": agent1_output,
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


def run_agent2(
    source_path: str,
    agent1_output_path: str,
    prompt_path: str,
) -> str:

    prompt = load_prompt(prompt_path)

    source_problem = load_source_problem(source_path)
    agent1_output = load_text(agent1_output_path)

    return call_agent2(
        system_prompt=prompt,
        source_question=source_problem.get("question"),
        agent1_output=agent1_output,
    )

def call_agent2_revision(
    system_prompt: str,
    current_puzzle: str,
    feedback: str,
) -> str:

    payload = {
        "CURRENT_PUZZLE": current_puzzle,
        "VALIDATOR_FEEDBACK": feedback,
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

def run_agent2_revision(
    puzzle_path: str,
    feedback: str,
    revision_prompt_path: str,
) -> str:

    revision_prompt = load_prompt(
        revision_prompt_path
    )

    current_puzzle = load_text(puzzle_path)

    return call_agent2_revision(
        system_prompt=revision_prompt,
        current_puzzle=current_puzzle,
        feedback=feedback,
    )