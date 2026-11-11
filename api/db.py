"""Exams and students' submissions, in SQLite. Students' names are personal data."""
import json
import os
import sqlite3
import threading

from engine.rag import STORAGE

PATH = os.getenv("DB_PATH", str(STORAGE / "grisz.db"))
os.makedirs(os.path.dirname(PATH), exist_ok=True)
db = sqlite3.connect(PATH, check_same_thread=False, timeout=30)  # shared by the api and the worker
db.row_factory = sqlite3.Row
db.execute("PRAGMA journal_mode=WAL")
db.executescript("""
CREATE TABLE IF NOT EXISTS exams (
  id TEXT PRIMARY KEY, title TEXT, pdf TEXT, mix TEXT,
  status TEXT,                     -- writing, review, published, closed, failed
  progress TEXT DEFAULT '', code TEXT UNIQUE,
  questions TEXT DEFAULT '[]', problems TEXT DEFAULT '[]', created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS submissions (
  id INTEGER PRIMARY KEY, exam_id TEXT, student TEXT, answers TEXT, marks TEXT,
  score INT, total INT, created TEXT DEFAULT CURRENT_TIMESTAMP);
""")
lock = threading.Lock()
JSON_COLUMNS = {"mix", "questions", "problems", "answers", "marks"}


def query(sql, *args):
    """Run one SQL statement. Lists and dicts are stored as JSON and come back decoded; rows come back as dicts."""
    args = [json.dumps(a) if isinstance(a, (list, dict)) else a for a in args]
    with lock, db:
        rows = db.execute(sql, args).fetchall()
    return [{k: json.loads(r[k]) if k in JSON_COLUMNS and r[k] is not None else r[k] for k in r.keys()} for r in rows]
