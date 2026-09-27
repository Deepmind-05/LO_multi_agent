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


def load_puzzle(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_agent3_input(
    source_problem: dict,
    puzzle: str,
) -> dict:

    return {
        "TARGET_DIFFICULTY": source_problem.get("difficulty"),
        "PUZZLE": puzzle,
    }


def call_agent3(
    system_prompt: str,
    source_problem: dict,
    puzzle: str,
) -> str:

    payload = build_agent3_input(
        source_problem=source_problem,
        puzzle=puzzle,
    )

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


def run_agent3(
    source_path: str,
    puzzle_path: str,
    prompt_path: str,
) -> str:

    prompt = load_prompt(prompt_path)

    source_problem = load_source_problem(source_path)
    puzzle = load_puzzle(puzzle_path)

    return call_agent3(
        system_prompt=prompt,
        source_problem=source_problem,
        puzzle=puzzle,
    )

def parse_validation(output: str) -> tuple[str, str]:
    lines = [line.strip() for line in output.splitlines()]

    verdict = None
    issues = []

    for i, line in enumerate(lines):
        if line == "VERDICT" and i + 1 < len(lines):
            verdict = lines[i + 1]

        if line == "ISSUES":
            for issue in lines[i + 1:]:
                if issue.startswith("- "):
                    issues.append(issue[2:].strip())

    if verdict not in {"PASS", "PASS_WITH_ISSUES", "FAIL"}:
        raise ValueError(
            f"Invalid Agent 3 verdict: {verdict!r}"
        )

    feedback = "\n".join(
        f"- {issue}"
        for issue in issues
    )

    return verdict, feedback