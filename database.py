import sqlite3
from pathlib import Path

DB_PATH = Path().resolve().parent / "diagnostics.db"

def connect_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                truck_model TEXT NOT NULL,
                error_code TEXT,
                symptoms TEXT,
                diagnosis TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_diagnostic(truck_model, error_code, symptoms, diagnosis):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO diagnostics
            (truck_model, error_code, symptoms, diagnosis)
            VALUES (?, ?, ?, ?)
        """, (truck_model, error_code, symptoms, diagnosis))

def read_diagnostics():
    with connect_db() as conn:
        return conn.execute("""
            SELECT * FROM diagnostics ORDER BY id DESC
        """).fetchall()