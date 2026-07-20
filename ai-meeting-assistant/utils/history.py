import os
import sqlite3
import json

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meetings.db"))

def get_db_connection():
    """
    Establishes and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database schema if it doesn't already exist.
    Performs migrations if necessary (e.g. adding the analysis_json column).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            backend TEXT NOT NULL,
            duration REAL,
            transcript_text TEXT NOT NULL,
            summary_markdown TEXT,
            segments_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if analysis_json column exists
    cursor.execute("PRAGMA table_info(meetings)")
    columns = [col[1] for col in cursor.fetchall()]
    if "analysis_json" not in columns:
        print("[Database] Migrating: Adding column 'analysis_json' to table 'meetings'...")
        cursor.execute("ALTER TABLE meetings ADD COLUMN analysis_json TEXT")
        conn.commit()
        
    conn.close()

def save_meeting(
    title: str,
    source: str,
    backend: str,
    duration: float,
    transcript_text: str,
    summary_markdown: str,
    segments: list,
    analysis: dict = None
) -> int:
    """
    Saves a processed meeting run to the database, including the structured analysis dict.
    Returns the ID of the newly inserted row.
    """
    init_db()  # Ensure DB is initialized and migrated
    conn = get_db_connection()
    cursor = conn.cursor()
    
    segments_str = json.dumps(segments)
    
    # Default format for analysis if missing
    if analysis is None:
        analysis = {
            "action_items": [],
            "decisions": [],
            "open_questions": []
        }
    analysis_str = json.dumps(analysis)
    
    cursor.execute("""
        INSERT INTO meetings (title, source, backend, duration, transcript_text, summary_markdown, segments_json, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (title, source, backend, duration, transcript_text, summary_markdown, segments_str, analysis_str))
    
    meeting_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return meeting_id

def list_meetings() -> list:
    """
    Returns a list of all meetings in the database, ordered by creation date descending.
    Excludes large text columns (transcript_text, segments_json, analysis_json) for efficiency.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, source, backend, duration, created_at
        FROM meetings
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to standard dict list
    return [dict(row) for row in rows]

def get_meeting(meeting_id: int) -> dict:
    """
    Retrieves a single meeting record by its ID, deserializing segments and analysis dicts.
    Returns None if not found.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, source, backend, duration, transcript_text, summary_markdown, segments_json, analysis_json, created_at
        FROM meetings
        WHERE id = ?
    """, (meeting_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    meeting = dict(row)
    
    # Deserialize segments JSON
    try:
        meeting["segments"] = json.loads(meeting["segments_json"]) if meeting["segments_json"] else []
    except Exception:
        meeting["segments"] = []
        
    # Deserialize analysis JSON
    try:
        meeting["analysis"] = json.loads(meeting["analysis_json"]) if meeting["analysis_json"] else {
            "action_items": [],
            "decisions": [],
            "open_questions": []
        }
    except Exception:
        meeting["analysis"] = {
            "action_items": [],
            "decisions": [],
            "open_questions": []
        }
        
    return meeting

def delete_meeting(meeting_id: int) -> bool:
    """
    Deletes a meeting from the database by ID.
    Returns True if successfully deleted, False otherwise.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()
    
    return deleted_rows > 0
