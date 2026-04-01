"""
scheduler.py — Analyso Background Task Manager
──────────────────────────────────────────────
Sets up APScheduler to run background tasks periodically.
"""

from apscheduler.schedulers.background import BackgroundScheduler

_scheduler = None

def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return

    print("🕒 Starting background scheduler...")
    _scheduler = BackgroundScheduler(daemon=True)
    
    _scheduler.start()
    print("✅ Scheduler running")
