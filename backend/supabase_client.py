"""
supabase_client.py — Analyso Supabase Connection Manager
─────────────────────────────────────────────────────────
Provides a singleton Supabase client used across the app.
Fails gracefully if env vars are not set — app still runs.
"""

import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_supabase_client = None

def get_supabase():
    """
    Returns a singleton Supabase client.
    Returns None if credentials are not configured —
    all callers must handle None gracefully.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠️  Supabase not configured — signal tracking disabled.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected successfully!")
        return _supabase_client
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return None
