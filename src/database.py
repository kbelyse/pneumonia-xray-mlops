"""SQLite tables for uploaded images and retraining runs."""
import json
import sqlite3
import time
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label TEXT NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    uploaded_at TEXT NOT NULL,
    retrain_run INTEGER
);
CREATE TABLE IF NOT EXISTS retrain_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    step TEXT,
    trigger TEXT,
    n_new INTEGER,
    n_replay INTEGER,
    old_metrics TEXT,
    new_metrics TEXT,
    promoted INTEGER,
    model_version INTEGER,
    message TEXT
);
"""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def add_upload(original_name, stored_path, label, sha) -> bool:
    with connect() as con:
        try:
            con.execute("INSERT INTO uploads (original_name, stored_path, label, sha256, uploaded_at) VALUES (?,?,?,?,?)",
                        (original_name, str(stored_path), label, sha, now()))
            return True
        except sqlite3.IntegrityError:  # same image uploaded before
            return False


def pending_uploads():
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads WHERE retrain_run IS NULL ORDER BY id")]


def upload_summary() -> dict:
    with connect() as con:
        rows = con.execute("""SELECT label, COUNT(*) AS total, SUM(retrain_run IS NULL) AS pending
                              FROM uploads GROUP BY label""").fetchall()
    return {r["label"]: {"total": r["total"], "pending": r["pending"]} for r in rows}


def mark_used(upload_ids, run_id):
    with connect() as con:
        con.executemany("UPDATE uploads SET retrain_run=? WHERE id=?", [(run_id, i) for i in upload_ids])


def start_run(trigger: str):
    """Returns the new run id, or None if a retrain is already running."""
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        busy = con.execute("SELECT id FROM retrain_runs WHERE status='running'").fetchone()
        if busy:
            return None
        cur = con.execute("INSERT INTO retrain_runs (started_at, status, step, trigger) VALUES (?,?,?,?)",
                          (now(), "running", "queued", trigger))
        return cur.lastrowid


def update_run(run_id, **fields):
    for k in ("old_metrics", "new_metrics"):
        if k in fields and not isinstance(fields[k], str):
            fields[k] = json.dumps(fields[k])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE retrain_runs SET {cols} WHERE id=?", (*fields.values(), run_id))


def recent_runs(limit: int = 10):
    with connect() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,))]
    for r in rows:
        for k in ("old_metrics", "new_metrics"):
            r[k] = json.loads(r[k]) if r[k] else None
    return rows


def fail_stale_runs():
    """A container that restarts mid-retrain leaves a 'running' row behind. Close it on startup."""
    with connect() as con:
        con.execute("UPDATE retrain_runs SET status='failed', message='container restarted', finished_at=? "
                    "WHERE status='running' AND started_at < datetime('now', 'localtime', '-2 hours')", (now(),))
