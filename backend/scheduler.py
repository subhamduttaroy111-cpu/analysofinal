"""
scheduler.py — Analyso Background Task Manager
──────────────────────────────────────────────
Sets up APScheduler to run background tasks periodically.
"""

from apscheduler.schedulers.background import BackgroundScheduler
import signal_tracker

_scheduler = None

def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return

    print("🕒 Starting background scheduler...")
    _scheduler = BackgroundScheduler(daemon=True)
    
    # Run the signal evaluator every 7 minutes
    _scheduler.add_job(
        signal_tracker.update_signal_results,
        'interval',
        minutes=7,
        id='update_signal_results_job',
        replace_existing=True
    )
    
    _scheduler.start()
    print("✅ Scheduler running (evaluating signals every 7 mins)")
