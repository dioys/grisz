"""Phase 6: quiz results per student, saved in storage/results.db (personal data: names)."""
import json
import sqlite3
import threading

from engine.rag import STORAGE

STORAGE.mkdir(exist_ok=True)
db = sqlite3.connect(STORAGE / "results.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS results "
           "(quiz_id TEXT, student TEXT, score INT, total INT, missed TEXT, at TEXT DEFAULT CURRENT_TIMESTAMP)")
lock = threading.Lock()  # one request at a time touches the database


def save(quiz_id, student, score, total, missed):
    with lock, db:
        db.execute("INSERT INTO results (quiz_id, student, score, total, missed) VALUES (?, ?, ?, ?, ?)",
                   (quiz_id, student, score, total, json.dumps(missed)))


def all_rows():
    with lock:
        return db.execute("SELECT quiz_id, student, score, total, missed, at FROM results ORDER BY at").fetchall()


def delete(student):
    """Removes every result for one student (any letter case). Returns how many were removed."""
    with lock, db:
        return db.execute("DELETE FROM results WHERE lower(student) = lower(?)", (student,)).rowcount
