from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from agent1.agent import run_agent1
from agent2.agent import run_agent2, run_agent2_revision
from agent3.agent import run_agent3, parse_validation


BASE_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    return BASE_DIR / path


def save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    config = load_config()

    paths = config["paths"]
    max_retries = config["max_retries"]

    source_path = resolve_path(paths["source"])
    languages_path = resolve_path(paths["languages"])

    agent1_prompt_path = resolve_path(paths["agent1_prompt"])
    agent1_output_path = resolve_path(paths["agent1_output"])

    agent2_prompt_path = resolve_path(paths["agent2_prompt"])
    agent2_revision_prompt_path = resolve_path(
        paths["agent2_revision_prompt"]
    )
    agent2_output_path = resolve_path(paths["agent2_output"])

    agent3_prompt_path = resolve_path(paths["agent3_prompt"])
    agent3_output_path = resolve_path(paths["agent3_output"])

    failed_puzzle_path = resolve_path(paths["failed_puzzle"])
    failed_validation_path = resolve_path(paths["failed_validation"])

    # --------------------------------------------------
    # AGENT 1
    # --------------------------------------------------

    print("Running Agent 1...")

    agent1_result = run_agent1(
        language_id=config["target_language_id"],
        source_path=str(source_path),
        languages_path=str(languages_path),
        prompt_path=str(agent1_prompt_path),
    )

    save_text(agent1_output_path, agent1_result)

    print("Agent 1 complete.")

    # --------------------------------------------------
    # AGENT 2 - INITIAL
    # --------------------------------------------------

    print("Running Agent 2...")

    puzzle = run_agent2(
        source_path=str(source_path),
        agent1_output_path=str(agent1_output_path),
        prompt_path=str(agent2_prompt_path),
    )

    save_text(agent2_output_path, puzzle)

    print("Agent 2 complete.")

    # --------------------------------------------------
    # AGENT 3 + REVISION LOOP
    # --------------------------------------------------

    for attempt in range(max_retries + 1):

        print(f"Running Agent 3 (attempt {attempt + 1})...")

        validation = run_agent3(
            source_path=str(source_path),
            puzzle_path=str(agent2_output_path),
            prompt_path=str(agent3_prompt_path),
        )

        save_text(agent3_output_path, validation)

        verdict, feedback = parse_validation(validation)

        print(f"Agent 3 verdict: {verdict}")

        # ----------------------------------------------
        # PASS
        # ----------------------------------------------

        if verdict == "PASS":
            print("Puzzle passed validation.")
            print(f"Final puzzle: {agent2_output_path}")
            return

        # ----------------------------------------------
        # RETRIES EXHAUSTED
        # ----------------------------------------------

        if attempt >= max_retries:
            print("Maximum retries reached.")
            print("Puzzle failed validation.")

            save_text(
                failed_puzzle_path,
                puzzle,
            )

            save_text(
                failed_validation_path,
                validation,
            )

            return

        # ----------------------------------------------
        # REVISION
        # ----------------------------------------------

        print("Sending feedback to Agent 2...")
        print(feedback)

        puzzle = run_agent2_revision(
            puzzle_path=str(agent2_output_path),
            feedback=feedback,
            revision_prompt_path=str(agent2_revision_prompt_path),
        )

        save_text(agent2_output_path, puzzle)

        print(f"Agent 2 revision {attempt + 1} complete.")


if __name__ == "__main__":
    main()