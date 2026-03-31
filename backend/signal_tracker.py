"""
signal_tracker.py — Analyso Signal Storage & Performance Analytics
────────────────────────────────────────────────────────────────────
Handles:
  - Saving signals to Supabase after every scan
  - Auto-evaluating WIN/LOSS by checking live prices
  - Accuracy & dashboard stats calculations
"""

import yfinance as yf
from datetime import datetime, timezone
from supabase_client import get_supabase

# ─────────────────────────────────────────────────────────
# Timeframe mapping
# ─────────────────────────────────────────────────────────
MODE_TIMEFRAME = {
    "INTRADAY":  "15m",
    "SWING":     "1d",
    "LONG_TERM": "weekly"
}

# ─────────────────────────────────────────────────────────
# 1. SAVE SIGNAL
# ─────────────────────────────────────────────────────────
def save_signal(stock: str, mode: str, score: int,
                price: float, target: float, stoploss: float,
                rr_ratio: float, bias: str) -> bool:
    """
    Insert one signal row into Supabase signals table.
    Skips if Supabase is not configured.
    Skips if the same stock+mode signal was saved in the last 5 minutes
    (prevents duplicate saves from cached scan results).
    """
    db = get_supabase()
    if db is None:
        return False

    try:
        strategy = mode.lower().replace("long_term", "longterm")
        signal   = "BUY" if bias == "BULLISH" else "SELL" if bias == "BEARISH" else "NEUTRAL"
        timeframe = MODE_TIMEFRAME.get(mode, "1d")

        # ── Duplicate guard: same stock+mode in last 5 min ──
        recent = (
            db.table("signals")
              .select("id, created_at")
              .eq("stock", stock)
              .eq("mode", mode)
              .eq("result", "OPEN")
              .order("created_at", desc=True)
              .limit(1)
              .execute()
        )
        if recent.data:
            last_ts = datetime.fromisoformat(recent.data[0]["created_at"].replace("Z", "+00:00"))
            diff = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if diff < 300:  # 5 minutes
                return False  # Skip duplicate

        row = {
            "stock":    stock,
            "signal":   signal,
            "price":    round(float(price), 2),
            "target":   round(float(target), 2),
            "stoploss": round(float(stoploss), 2),
            "timeframe": timeframe,
            "strategy": strategy,
            "mode":     mode,
            "score":    int(score),
            "rr_ratio": round(float(rr_ratio), 2),
            "result":   "OPEN",
            "profit_pct": 0.0,
        }

        db.table("signals").insert(row).execute()
        print(f"📊 Signal saved: {stock} {signal} @ ₹{price}")
        return True

    except Exception as e:
        print(f"⚠️  save_signal error ({stock}): {e}")
        return False


# ─────────────────────────────────────────────────────────
# 2. UPDATE SIGNAL RESULTS (WIN / LOSS evaluator)
# ─────────────────────────────────────────────────────────
def update_signal_results() -> dict:
    """
    Fetch all OPEN signals from Supabase.
    For each, get the current live price via yfinance.
    Evaluate:
      - current_price >= target  → WIN
      - current_price <= stoploss → LOSS
      - else                     → stays OPEN
    Updates rows in Supabase in bulk.
    Returns dict with counts.
    """
    db = get_supabase()
    if db is None:
        return {"updated": 0, "wins": 0, "losses": 0}

    try:
        response = (
            db.table("signals")
              .select("id, stock, price, target, stoploss, signal")
              .eq("result", "OPEN")
              .execute()
        )
        open_signals = response.data

        if not open_signals:
            return {"updated": 0, "wins": 0, "losses": 0}

        print(f"🔄 Evaluating {len(open_signals)} open signals...")

        # Get unique stocks to fetch prices once
        unique_stocks = list({s["stock"] for s in open_signals})
        symbols = [f"{s}.NS" for s in unique_stocks]

        # Batch download current prices
        prices = {}
        try:
            data = yf.download(symbols, period="1d", interval="5m",
                               group_by="ticker", progress=False, threads=True)
            for sym in unique_stocks:
                try:
                    ns = f"{sym}.NS"
                    if len(unique_stocks) == 1:
                        col_data = data["Close"]
                    else:
                        col_data = data[ns]["Close"]
                    prices[sym] = float(col_data.dropna().iloc[-1])
                except Exception:
                    prices[sym] = None
        except Exception as e:
            print(f"⚠️  Price fetch error: {e}")
            return {"updated": 0, "wins": 0, "losses": 0}

        wins = losses = 0

        for sig in open_signals:
            stock    = sig["stock"]
            entry    = float(sig["price"])
            target   = float(sig["target"])
            stoploss = float(sig["stoploss"])
            signal   = sig["signal"]  # BUY or SELL
            current  = prices.get(stock)

            if current is None:
                continue

            # Evaluate result
            result = "OPEN"
            if signal == "BUY":
                if current >= target:
                    result = "WIN"
                elif current <= stoploss:
                    result = "LOSS"
            elif signal == "SELL":
                if current <= target:
                    result = "WIN"
                elif current >= stoploss:
                    result = "LOSS"

            if result == "OPEN":
                continue

            # Calculate profit %
            if signal == "BUY":
                profit_pct = ((current - entry) / entry) * 100
            else:
                profit_pct = ((entry - current) / entry) * 100

            # Update in Supabase
            try:
                db.table("signals").update({
                    "result":     result,
                    "profit_pct": round(profit_pct, 2)
                }).eq("id", sig["id"]).execute()

                if result == "WIN":
                    wins += 1
                else:
                    losses += 1

                print(f"  ✅ {stock}: {result} ({profit_pct:+.1f}%)")
            except Exception as e:
                print(f"  ⚠️  Update error for {stock}: {e}")

        summary = {"updated": wins + losses, "wins": wins, "losses": losses}
        print(f"🏁 Update complete: {wins} wins, {losses} losses")
        return summary

    except Exception as e:
        print(f"❌ update_signal_results error: {e}")
        return {"updated": 0, "wins": 0, "losses": 0}


# ─────────────────────────────────────────────────────────
# 3. CALCULATE ACCURACY
# ─────────────────────────────────────────────────────────
def calculate_accuracy(strategy: str = None) -> dict:
    """
    Returns accuracy statistics.
    strategy: 'swing' | 'longterm' | 'intraday' | None (all)
    """
    db = get_supabase()
    if db is None:
        return {"accuracy": 0, "wins": 0, "losses": 0, "total": 0}

    try:
        query = db.table("signals").select("result, strategy").neq("result", "OPEN")

        if strategy:
            query = query.eq("strategy", strategy)

        response = query.execute()
        rows = response.data

        total = len(rows)
        wins  = sum(1 for r in rows if r["result"] == "WIN")
        losses = total - wins
        accuracy = round((wins / total) * 100, 1) if total > 0 else 0.0

        return {
            "strategy": strategy or "all",
            "accuracy": accuracy,
            "wins": wins,
            "losses": losses,
            "total": total
        }

    except Exception as e:
        print(f"❌ calculate_accuracy error: {e}")
        return {"accuracy": 0, "wins": 0, "losses": 0, "total": 0}


# ─────────────────────────────────────────────────────────
# 4. FULL DASHBOARD STATS
# ─────────────────────────────────────────────────────────
def get_dashboard_stats() -> dict:
    """
    Returns complete stats for the performance dashboard.
    """
    db = get_supabase()
    if db is None:
        return _empty_stats()

    try:
        all_resp  = db.table("signals").select("result, profit_pct, strategy").execute()
        rows = all_resp.data

        total   = len(rows)
        open_c  = sum(1 for r in rows if r["result"] == "OPEN")
        wins    = sum(1 for r in rows if r["result"] == "WIN")
        losses  = sum(1 for r in rows if r["result"] == "LOSS")
        closed  = wins + losses

        accuracy = round((wins / closed) * 100, 1) if closed > 0 else 0.0

        closed_profits = [r["profit_pct"] for r in rows if r["result"] != "OPEN" and r["profit_pct"] is not None]
        avg_profit = round(sum(closed_profits) / len(closed_profits), 2) if closed_profits else 0.0

        # Strategy-wise breakdown
        strategies = ["swing", "longterm", "intraday"]
        strategy_stats = {}
        for s in strategies:
            s_rows = [r for r in rows if r.get("strategy") == s]
            s_closed = [r for r in s_rows if r["result"] != "OPEN"]
            s_wins = sum(1 for r in s_closed if r["result"] == "WIN")
            s_total = len(s_closed)
            strategy_stats[s] = {
                "total": len(s_rows),
                "wins":  s_wins,
                "losses": s_total - s_wins,
                "accuracy": round((s_wins / s_total) * 100, 1) if s_total > 0 else 0.0
            }

        return {
            "total_trades": total,
            "open_trades":  open_c,
            "wins":         wins,
            "losses":       losses,
            "closed_trades": closed,
            "accuracy":     accuracy,
            "avg_profit_pct": avg_profit,
            "by_strategy":  strategy_stats
        }

    except Exception as e:
        print(f"❌ get_dashboard_stats error: {e}")
        return _empty_stats()


def _empty_stats():
    return {
        "total_trades": 0, "open_trades": 0,
        "wins": 0, "losses": 0, "closed_trades": 0,
        "accuracy": 0.0, "avg_profit_pct": 0.0,
        "by_strategy": {
            "swing":    {"total": 0, "wins": 0, "losses": 0, "accuracy": 0.0},
            "longterm": {"total": 0, "wins": 0, "losses": 0, "accuracy": 0.0},
            "intraday": {"total": 0, "wins": 0, "losses": 0, "accuracy": 0.0},
        }
    }
