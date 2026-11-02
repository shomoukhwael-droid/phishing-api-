import sqlite3
import hashlib
import os
import secrets
from datetime import datetime
from contextlib import contextmanager

DB_NAME = "amn.db"

@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_developer INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS checked_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                email_content TEXT NOT NULL,
                sender_address TEXT,
                is_phishing INTEGER NOT NULL,
                confidence REAL,
                source_note TEXT,
                checked_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        # جدول الروابط المخصص لحفظ نتائج تحليل الروابط
        cur.execute("""
            CREATE TABLE IF NOT EXISTS checked_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                url_content TEXT NOT NULL,
                is_phishing INTEGER NOT NULL,
                confidence REAL,
                checked_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                theme TEXT NOT NULL DEFAULT 'dark',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        print("done creating tables")

def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    ).hex()
    return pw_hash, salt

def create_user(username, email, password, is_developer=False):
    pw_hash, salt = _hash_password(password)
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (username, email, password_hash, salt, is_developer, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, email, pw_hash, salt, int(is_developer), now))
        except sqlite3.IntegrityError:
            raise ValueError("username or email already exists")
        user_id = cur.lastrowid
        cur.execute("INSERT INTO user_settings (user_id, theme) VALUES (?, 'dark')", (user_id,))
        return user_id

def verify_user(username_or_email, password):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ? OR email = ?",
                     (username_or_email, username_or_email))
        row = cur.fetchone()
        if row is None:
            return None
        pw_hash, _ = _hash_password(password, row["salt"])
        if pw_hash != row["password_hash"]:
            return None
        return dict(row)

def save_checked_email(user_id, email_content, is_phishing, confidence=None,
                       sender_address=None, source_note=None):
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO checked_emails
                (user_id, email_content, sender_address, is_phishing, confidence, source_note, checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, email_content, sender_address, int(is_phishing), confidence, source_note, now))
        return cur.lastrowid

def save_checked_url(user_id, url_content, is_phishing, confidence=None):
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO checked_urls
                (user_id, url_content, is_phishing, confidence, checked_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, url_content, int(is_phishing), confidence, now))
        return cur.lastrowid

def get_email_history(user_id, limit=50):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM checked_emails WHERE user_id = ?
            ORDER BY checked_at DESC LIMIT ?
        """, (user_id, limit))
        return [dict(r) for r in cur.fetchall()]

def get_url_history(user_id, limit=50):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM checked_urls WHERE user_id = ?
            ORDER BY checked_at DESC LIMIT ?
        """, (user_id, limit))
        return [dict(r) for r in cur.fetchall()]

def get_user_theme(user_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT theme FROM user_settings WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        return row["theme"] if row else "dark"

def update_user_theme(user_id, theme):
    if theme not in ("dark", "light"):
        raise ValueError("theme must be dark or light")
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_settings (user_id, theme) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET theme = excluded.theme
        """, (user_id, theme))

def submit_complaint(user_id, message):
    now = datetime.now().isoformat()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO complaints (user_id, message, status, created_at)
            VALUES (?, ?, 'open', ?)
        """, (user_id, message, now))
        return cur.lastrowid

def get_open_complaints():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT complaints.*, users.username, users.email
            FROM complaints JOIN users ON users.id = complaints.user_id
            WHERE status = 'open' ORDER BY created_at DESC
        """)
        return [dict(r) for r in cur.fetchall()]