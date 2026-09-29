import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "experiments.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database and ensure puzzle_runs table exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS puzzle_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            question_id TEXT NOT NULL,
            original_language TEXT,
            target_language TEXT NOT NULL,
            target_concepts TEXT,
            target_rationale TEXT,
            target_puzzle TEXT,
            validation_json TEXT,
            verdict TEXT NOT NULL,
            target_difficulty INTEGER,
            actual_difficulty INTEGER,
            attempts_used INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            output_dir TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_name ON puzzle_runs(run_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_id ON puzzle_runs(question_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON puzzle_runs(verdict)")
    conn.commit()
    return conn


def log_puzzle_run(
    run_name: str,
    question_id: str,
    original_language: str | None,
    target_language: str,
    target_concepts: str | None,
    target_rationale: str | None,
    target_puzzle: str | None,
    validation_data: dict | str | None,
    verdict: str,
    target_difficulty: int | None,
    actual_difficulty: int | None,
    attempts_used: int,
    elapsed_seconds: float,
    output_dir: str,
    db_path: Path = DB_PATH,
) -> int:
    """Inserts a completed puzzle run into the SQLite database."""
    conn = init_db(db_path)
    val_json_str = (
        json.dumps(validation_data, ensure_ascii=False)
        if isinstance(validation_data, dict)
        else (validation_data or "")
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    with conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO puzzle_runs (
                run_name,
                created_at,
                question_id,
                original_language,
                target_language,
                target_concepts,
                target_rationale,
                target_puzzle,
                validation_json,
                verdict,
                target_difficulty,
                actual_difficulty,
                attempts_used,
                elapsed_seconds,
                output_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                now_iso,
                question_id,
                original_language,
                target_language,
                target_concepts,
                target_rationale,
                target_puzzle,
                val_json_str,
                verdict,
                target_difficulty,
                actual_difficulty,
                attempts_used,
                round(elapsed_seconds, 2),
                output_dir,
            ),
        )
        record_id = cursor.lastrowid

    conn.close()
    return record_id
