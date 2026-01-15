import sqlite3
from pathlib import Path

DB_PATH = Path("/app/data/results.db")

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT,
            ok INTEGER,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_result(result: dict):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO results (plate, ok, reason) VALUES (?, ?, ?)",
        (
            result.get("plate"),
            int(result.get("ok", False)),
            result.get("reason"),
        ),
    )

    conn.commit()
    conn.close()
