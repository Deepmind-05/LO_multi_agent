#!/usr/bin/env python3
"""
Automated Pipeline to Build source.jsonl from Linguistic Olympiad Cleaned Data.

Pipeline Stages:
1. Read problem from input JSONL (e.g., cleaned_with_difficulty.jsonl).
2. Call Agent 1 (Concept Architect) to extract concepts, WALS features, and source language.
3. Call Agent 2 (Rationale Architect) to construct the problem-setting rationale.
4. Python Algorithmic Matcher: Searches languages.jsonl to find Top 10 matching candidate languages.
5. Call Agent 3 (Language Rarity Filter) to evaluate Top 10 and select Top 5 rare/extinct/less spoken languages.
6. Assemble into source.json schema and append to source.jsonl incrementally (resumable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# =====================================================================
# Configuration & Helpers
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_FILE = Path(r"D:\Projects\Linguistic_Olympiad\data\cleaned_with_difficulty.jsonl")
DEFAULT_LANGUAGES_FILE = BASE_DIR / "data" / "languages.jsonl"
DEFAULT_OUTPUT_FILE = BASE_DIR / "data" / "source.jsonl"

PROMPT_CONCEPT_FILE = BASE_DIR / "prompts" / "concept_prompt.txt"
PROMPT_RATIONALE_FILE = BASE_DIR / "prompts" / "rationale_prompt.txt"
PROMPT_TARGET_LANG_FILE = BASE_DIR / "prompts" / "target_language_prompt.txt"

# Difficulty mapping:
# Breakthrough  → 1
# Foundation    → 2
# Intermediate  → 3
# Advanced      → 4
# Round 2       → 5
DIFFICULTY_MAP = {
    "breakthrough": 1,
    "foundation": 2,
    "intermediate": 3,
    "advanced": 4,
    "round 2": 5,
    "round2": 5,
}


def map_difficulty(raw_difficulty: Any, default: int = 4) -> int:
    """Map textual difficulty to integer 1-5."""
    if raw_difficulty is None:
        return default
    if isinstance(raw_difficulty, int):
        return raw_difficulty
    norm = str(raw_difficulty).strip().lower()
    return DIFFICULTY_MAP.get(norm, default)


def load_env(env_path: Path | None = None) -> dict[str, str]:
    """Parse .env file without external dependencies."""
    env_vars = {}
    search_paths = [
        env_path,
        BASE_DIR / ".env",
        Path.cwd() / ".env",
        Path(r"D:\Projects\LO_multi_agent_v2\.env"),
        Path(r"D:\Projects\LO_multi_agent\.env"),
    ]
    for p in search_paths:
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
                        env_vars[k] = v
            break
    return env_vars


def read_prompt(path: Path) -> str:
    """Read prompt text, ignoring pure comment lines."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines).strip()


def parse_json_from_response(content: str, fallback_extractor: Any = None) -> Any:
    """Safely extract JSON object or array from LLM response with multi-stage fallbacks."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Bracket range extraction
    first_curly = cleaned.find("{")
    last_curly = cleaned.rfind("}")
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")

    if first_curly != -1 and last_curly > first_curly:
        if first_bracket == -1 or first_curly < first_bracket:
            try:
                return json.loads(cleaned[first_curly : last_curly + 1])
            except json.JSONDecodeError:
                pass

    if first_bracket != -1 and last_bracket > first_bracket:
        try:
            return json.loads(cleaned[first_bracket : last_bracket + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: Scan string for embedded JSON object using raw_decode
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(cleaned):
        while pos < len(cleaned) and cleaned[pos] not in "{[":
            pos += 1
        if pos >= len(cleaned):
            break
        try:
            obj, _ = decoder.raw_decode(cleaned, pos)
            return obj
        except Exception:
            pos += 1

    # Strategy 4: Fallback heuristic extractor if conversational text was returned
    if fallback_extractor and callable(fallback_extractor):
        return fallback_extractor(content)

    raise ValueError(f"Could not parse valid JSON from response: {content[:300]}...")


# =====================================================================
# Zero-Dependency LLM Client (Urllib with Cloudflare Bypass)
# =====================================================================

class SimpleLLMClient:
    """Zero-dependency HTTP client supporting Cerebras, Groq, OpenAI, and Gemini."""

    def __init__(
        self,
        provider: str = "cerebras",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.2,
    ):
        load_env()
        self.provider = provider.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.provider == "cerebras":
            self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
            self.model = model or "qwen-3.8-27b"
            self.endpoint = "https://api.cerebras.ai/v1/chat/completions"
        elif self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.model = model or "llama-3.3-70b-versatile"
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        elif self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            self.model = model or "gemini-2.5-flash"
            self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"
            self.endpoint = "https://api.openai.com/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        if not self.api_key:
            raise ValueError(
                f"Missing API key for provider '{self.provider}'. "
                f"Please ensure {self.provider.upper()}_API_KEY is set in your .env or environment."
            )

    def complete(
        self,
        system_prompt: str,
        user_payload: str | dict,
        max_retries: int = 5,
        json_mode: bool = False,
    ) -> str:
        """Send chat completion with exponential backoff on rate limits."""
        if isinstance(user_payload, (dict, list)):
            user_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        else:
            user_text = str(user_payload)

        # Common headers to avoid Cloudflare 403 (error code 1010)
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        if self.provider == "gemini":
            url = f"{self.endpoint}?key={self.api_key}"
            body_dict = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": {
                    "maxOutputTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            }
            if json_mode:
                body_dict["generationConfig"]["responseMimeType"] = "application/json"
            headers = {"Content-Type": "application/json", "User-Agent": ua}
        else:
            url = self.endpoint
            body_dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "max_completion_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.provider == "cerebras":
                body_dict["reasoning_effort"] = "low"
            if json_mode:
                body_dict["response_format"] = {"type": "json_object"}

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": ua,
            }

        body_data = json.dumps(body_dict).encode("utf-8")

        for attempt in range(max_retries + 1):
            req = urllib.request.Request(url, data=body_data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))

                if self.provider == "gemini":
                    return resp_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    msg = resp_json["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and "reasoning" in msg:
                        content = msg["reasoning"]
                    return content

            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="ignore")
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait_sec = min(2 ** (attempt + 1) * 3, 60)
                    print(f"  [LLM Warning] HTTP {e.code}: Retrying in {wait_sec}s... ({err_text[:100]})")
                    time.sleep(wait_sec)
                else:
                    raise RuntimeError(f"LLM API Error ({e.code}): {err_text}") from e
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    raise e

        return ""


# =====================================================================
# WALS Algorithmic Matching (Top 10)
# =====================================================================

def load_wals_languages(languages_file: Path) -> list[dict]:
    """Load preprocessed WALS languages dataset."""
    languages = []
    with open(languages_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                languages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return languages


def find_top_10_wals_languages(
    required_features: list[dict],
    source_language: str,
    wals_languages: list[dict],
) -> list[dict]:
    """
    Python code to match target languages:
    1. Compares required (feature_id, value_id) pairs against WALS languages.
    2. Excludes the source language.
    3. Ranks candidates by matched feature count (descending) and total documented features.
    4. Backfills to ensure Top 10 candidate languages are always returned.
    """
    feature_pairs = {
        (str(f.get("feature_id")).strip(), str(f.get("value_id")).strip())
        for f in required_features
        if f.get("feature_id") and f.get("value_id")
    }

    source_norm = source_language.strip().casefold()
    scored = []

    for lang in wals_languages:
        lang_id = str(lang.get("language_id", "")).strip().casefold()
        lang_name = str(lang.get("language_name", "")).strip().casefold()

        # Exclude source language
        if source_norm and (source_norm == lang_name or source_norm == lang_id or source_norm in lang_name):
            continue

        # Extract features for this language
        lang_features = {
            (str(f.get("feature_id")).strip(), str(f.get("value_id")).strip())
            for f in lang.get("features", [])
        }

        matched_pairs = feature_pairs & lang_features
        matched_count = len(matched_pairs)
        total_feats = len(lang.get("features", []))

        # Only consider languages with at least 1 feature match if required features exist
        if feature_pairs and matched_count == 0:
            continue

        scored.append({
            "language_id": lang.get("language_id"),
            "language_name": lang.get("language_name"),
            "family": lang.get("family", ""),
            "genus": lang.get("genus", ""),
            "macroarea": lang.get("macroarea", ""),
            "matched_feature_count": matched_count,
            "total_feature_count": total_feats,
            "matched_features": [
                {"feature_id": fid, "value_id": vid}
                for fid, vid in sorted(matched_pairs)
            ],
            "raw_features_summary": [
                {"feature_id": f.get("feature_id"), "feature_name": f.get("feature_name"), "value_name": f.get("value_name")}
                for f in lang.get("features", [])[:10]
            ],
        })

    # Sort primarily by matched feature count (descending), secondarily by total features (descending)
    scored.sort(
        key=lambda x: (x["matched_feature_count"], x["total_feature_count"]),
        reverse=True,
    )

    # If fewer than 10 languages matched, backfill with diverse well-documented WALS languages
    if len(scored) < 10:
        seen_ids = {s["language_id"] for s in scored}
        fallback_langs = []
        for lang in wals_languages:
            lid = lang.get("language_id")
            lname = str(lang.get("language_name", "")).strip().casefold()
            if lid in seen_ids or (source_norm and (source_norm == lname or source_norm in lname)):
                continue
            fallback_langs.append({
                "language_id": lid,
                "language_name": lang.get("language_name"),
                "family": lang.get("family", ""),
                "genus": lang.get("genus", ""),
                "macroarea": lang.get("macroarea", ""),
                "matched_feature_count": 0,
                "total_feature_count": len(lang.get("features", [])),
                "matched_features": [],
                "raw_features_summary": [
                    {"feature_id": f.get("feature_id"), "feature_name": f.get("feature_name"), "value_name": f.get("value_name")}
                    for f in lang.get("features", [])[:10]
                ],
            })
        fallback_langs.sort(key=lambda x: x["total_feature_count"], reverse=True)
        scored.extend(fallback_langs[:(10 - len(scored))])

    # Return top 10 candidates
    return scored[:10]


# =====================================================================
# Upstream Agents
# =====================================================================

def run_concept_agent(
    client: SimpleLLMClient,
    prompt_template: str,
    problem_data: dict,
    question_id: str,
) -> dict:
    """Agent 1: Extracts concepts, WALS features, and source language."""
    if not prompt_template:
        raise ValueError(
            "Concept prompt is empty! Please paste your prompt into prompts/concept_prompt.txt"
        )

    def _heuristic_concept_fallback(text: str) -> dict:
        lang_match = re.search(r"\b(?:about|in|language of|language is)\s+([A-Z][a-z]+)", text)
        lang = lang_match.group(1) if lang_match else ""
        concepts = []
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^(\d+\.|\*|-)\s+\*\*", line):
                c_name = re.sub(r"^(\d+\.|\*|-)\s+\*\*([^*]+)\*\*.*", r"\2", line).strip()
                concepts.append({"feature_id": None, "feature_name": c_name, "value_id": None, "value_name": line})
        return {
            "question_id": question_id,
            "source_language": lang,
            "difficulty": 3,
            "features": concepts,
            "tested_concepts_summary": text[:200],
        }

    response_text = client.complete(
        system_prompt=prompt_template,
        user_payload={
            "QUESTION_ID": question_id,
            "PREAMBLE": problem_data.get("preamble", ""),
            "CONTEXT": problem_data.get("context", ""),
            "QUESTIONS": problem_data.get("questions", []),
        },
        json_mode=True,
    )

    parsed = parse_json_from_response(response_text, fallback_extractor=_heuristic_concept_fallback)
    if isinstance(parsed, dict):
        return parsed
    return {"raw_response": response_text}


def run_rationale_agent(
    client: SimpleLLMClient,
    prompt_template: str,
    problem_data: dict,
    concepts_data: dict,
) -> str:
    """Agent 2: Constructs the problem construction rationale."""
    if not prompt_template:
        raise ValueError(
            "Rationale prompt is empty! Please paste your prompt into prompts/rationale_prompt.txt"
        )

    response_text = client.complete(
        system_prompt=prompt_template,
        user_payload={
            "QUESTION": {
                "preamble": problem_data.get("preamble", ""),
                "context": problem_data.get("context", ""),
                "questions": problem_data.get("questions", []),
            },
            "CONCEPTS": concepts_data,
        },
        json_mode=False,
    )
    return response_text.strip()


def run_target_language_agent(
    client: SimpleLLMClient,
    prompt_template: str,
    top_10_candidates: list[dict],
    problem_summary: dict,
) -> list[str]:
    """Agent 3: Evaluates Top 10 candidate languages and selects Top 5 rare/extinct/less spoken languages."""
    if not prompt_template:
        raise ValueError(
            "Target language prompt is empty! Please paste your prompt into prompts/target_language_prompt.txt"
        )

    response_text = client.complete(
        system_prompt=prompt_template,
        user_payload={
            "PROBLEM_SUMMARY": problem_summary,
            "TOP_10_CANDIDATE_LANGUAGES": top_10_candidates,
        },
        json_mode=True,
    )

    parsed = parse_json_from_response(response_text)

    # Extract list of language IDs
    selected_ids = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                selected_ids.append(item.strip())
            elif isinstance(item, dict) and "language_id" in item:
                selected_ids.append(item["language_id"].strip())
    elif isinstance(parsed, dict):
        for key in ["target_languages", "selected_languages", "rare_languages", "languages"]:
            if key in parsed and isinstance(parsed[key], list):
                for item in parsed[key]:
                    if isinstance(item, str):
                        selected_ids.append(item.strip())
                    elif isinstance(item, dict) and "language_id" in item:
                        selected_ids.append(item["language_id"].strip())
                break

    # If parsing returned less than 5, fallback to candidate order
    if not selected_ids:
        selected_ids = [c["language_id"] for c in top_10_candidates[:5]]

    return selected_ids[:5]


# =====================================================================
# Main Pipeline Runner
# =====================================================================

def build_source_dataset(
    input_file: Path,
    output_file: Path,
    languages_file: Path,
    provider: str,
    model: str | None,
    start_index: int = 0,
    limit: int | None = None,
    resume: bool = True,
):
    print("=" * 70)
    print("LO Source Dataset Generator (Upstream Multi-Agent Pipeline)")
    print("=" * 70)
    print(f"Input file:      {input_file}")
    print(f"Output file:     {output_file}")
    print(f"Languages file:  {languages_file}")
    print(f"LLM Provider:    {provider} (Model: {model or 'default'})")
    print(f"Resume enabled:  {resume}")
    print()

    # Load prompts
    concept_prompt = read_prompt(PROMPT_CONCEPT_FILE)
    rationale_prompt = read_prompt(PROMPT_RATIONALE_FILE)
    target_lang_prompt = read_prompt(PROMPT_TARGET_LANG_FILE)

    if not concept_prompt:
        print(f"[!] Warning: {PROMPT_CONCEPT_FILE} is empty. Fill it before running.")
    if not rationale_prompt:
        print(f"[!] Warning: {PROMPT_RATIONALE_FILE} is empty. Fill it before running.")
    if not target_lang_prompt:
        print(f"[!] Warning: {PROMPT_TARGET_LANG_FILE} is empty. Fill it before running.")

    if not concept_prompt or not rationale_prompt or not target_lang_prompt:
        print("\nPlease paste your prompt contents into the respective prompt files in:")
        print(f"  - {PROMPT_CONCEPT_FILE}")
        print(f"  - {PROMPT_RATIONALE_FILE}")
        print(f"  - {PROMPT_TARGET_LANG_FILE}")
        print("\nExiting until prompts are configured.")
        return

    # Initialize LLM client
    client = SimpleLLMClient(provider=provider, model=model)

    # Load WALS languages database
    print("Loading WALS languages database...")
    wals_languages = load_wals_languages(languages_file)
    print(f"Loaded {len(wals_languages)} WALS language profiles.\n")

    # Read existing records for resuming
    processed_ids = set()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace():
                    pos += 1
                if pos >= len(content):
                    break
                try:
                    obj, end = decoder.raw_decode(content, pos)
                    if "id" in obj:
                        processed_ids.add(str(obj["id"]))
                    pos = end
                except Exception:
                    pos += 1
        if processed_ids:
            print(f"Found {len(processed_ids)} already processed problems in {output_file.name}. Resuming...")

    # Load input problems
    problems = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    problems.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total_problems = len(problems)
    print(f"Total problems in input: {total_problems}")

    # Slice start and limit
    end_index = total_problems if limit is None else min(start_index + limit, total_problems)
    selected_problems = list(enumerate(problems))[start_index:end_index]

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, item in selected_problems:
            # Canonical unique ID strictly based on 1-indexed row number
            question_id = f"question_{idx + 1}"
            if resume and question_id in processed_ids:
                print(f"[{idx + 1}/{total_problems}] Skipping already processed {question_id}")
                continue

            print("-" * 70)
            print(f"[{idx + 1}/{total_problems}] Processing {question_id}...")

            # 1. Concept Extraction Agent
            print("  -> Running Concept Architect Agent...")
            concepts_data = run_concept_agent(
                client=client,
                prompt_template=concept_prompt,
                problem_data=item,
                question_id=question_id,
            )

            # Determine source language
            source_language = (
                concepts_data.get("source_language")
                or item.get("language")
                or ""
            )

            # Extract features for matching
            features_list = concepts_data.get("features", [])
            if not isinstance(features_list, list):
                features_list = []

            print(f"  -> Identified Source Language: '{source_language}' with {len(features_list)} WALS features.")

            # 2. Rationale Agent
            print("  -> Running Rationale Architect Agent...")
            rationale_text = run_rationale_agent(
                client=client,
                prompt_template=rationale_prompt,
                problem_data=item,
                concepts_data=concepts_data,
            )

            # 3. Python WALS Matching (Top 10)
            print("  -> Matching Top 10 Target Languages in WALS database...")
            top_10_candidates = find_top_10_wals_languages(
                required_features=features_list,
                source_language=source_language,
                wals_languages=wals_languages,
            )
            top_10_names = [f"{c['language_name']} ({c['language_id']})" for c in top_10_candidates]
            print(f"     Top 10 candidates: {top_10_names}")

            # 4. Target Language Rarity / Endangerment Filter Agent (Top 5)
            print("  -> Running Language Rarity Filter Agent (Top 5 rare/extinct)...")
            problem_summary = {
                "question_id": question_id,
                "source_language": source_language,
                "concepts": features_list,
            }
            top_5_target_languages = run_target_language_agent(
                client=client,
                prompt_template=target_lang_prompt,
                top_10_candidates=top_10_candidates,
                problem_summary=problem_summary,
            )
            print(f"  -> Selected Top 5 Target Languages: {top_5_target_languages}")

            # 5. Determine difficulty (Breakthrough: 1, Foundation: 2, Intermediate: 3, Advanced: 4, Round 2: 5)
            raw_diff = item.get("difficulty")
            if raw_diff is None:
                raw_diff = concepts_data.get("difficulty")
            difficulty = map_difficulty(raw_diff, default=4)
            print(f"  -> Difficulty mapped: '{raw_diff}' -> {difficulty}")

            # 6. Assemble complete record matching source.json schema + target_languages
            full_record = {
                "id": question_id,
                "question": {
                    "preamble": item.get("preamble", ""),
                    "context": item.get("context", ""),
                    "questions": item.get("questions", []),
                },
                "language": source_language,
                "rationale": rationale_text,
                "concepts": {
                    "question_id": question_id,
                    "source_language": source_language,
                    "features": features_list,
                },
                "difficulty": difficulty,
                "target_languages": top_5_target_languages,
            }

            # Write record immediately & flush
            out_f.write(json.dumps(full_record, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_ids.add(question_id)
            print(f"  [DONE] Saved {question_id} to {output_file.name}")

    print("\n" + "=" * 70)
    print(f"Processing Complete! Output saved to: {output_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build source.jsonl dataset from Linguistic Olympiad cleaned data."
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to input cleaned JSONL file (default: cleaned_with_difficulty.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output source.jsonl file",
    )
    parser.add_argument(
        "--languages", "-l",
        type=Path,
        default=DEFAULT_LANGUAGES_FILE,
        help="Path to WALS languages.jsonl",
    )
    parser.add_argument(
        "--provider", "-p",
        default="cerebras",
        choices=["cerebras", "groq", "gemini", "openai"],
        help="LLM Provider",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM Model Name",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start problem index (0-based)",
    )
    parser.add_argument(
        "--num-rows", "--rows", "-n", "--limit",
        dest="num_rows",
        type=int,
        default=None,
        help="Number of rows/problems to process (e.g. 5, 10, or all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (do not skip already processed problems)",
    )

    args = parser.parse_args()

    build_source_dataset(
        input_file=args.input,
        output_file=args.output,
        languages_file=args.languages,
        provider=args.provider,
        model=args.model,
        start_index=args.start,
        limit=args.num_rows,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()