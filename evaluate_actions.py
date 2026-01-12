#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Optional, Literal, Dict, Any, List, Tuple

import pytz
import pandas as pd
import yfinance as yf
import exchange_calendars as xcals
from supabase import create_client
from dotenv import load_dotenv


# ----------------------------
# Config
# ----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_ANON_KEY in env.", file=sys.stderr)
    sys.exit(1)

# Tables
ACTIONS_TABLE = os.getenv("ACTIONS_TABLE", "agent_actions")
EVALS_TABLE = os.getenv("EVALS_TABLE", "agent_evals")

# Horizons
HORIZONS: List[Literal["1d", "7d"]] = ["1d", "7d"]
HORIZON_MODE: Literal["calendar", "trading"] = os.getenv("HORIZON_MODE", "calendar")  # "calendar" or "trading"

# Market calendar
NY = pytz.timezone("America/New_York")
UTC = pytz.UTC
CAL = xcals.get_calendar("XNYS")  # US equities sessions/holidays

# yfinance settings
YF_TZ = NY  # yfinance daily bars align to exchange timezone


# ----------------------------
# Helpers: time + market close snapping
# ----------------------------
def parse_dt(value: Any) -> datetime:
    """
    Accepts ISO string or datetime.
    Returns timezone-aware UTC datetime.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        # pandas handles many ISO variants
        dt = pd.to_datetime(value).to_pydatetime()
    else:
        raise ValueError(f"Unsupported datetime type: {type(value)}")

    if dt.tzinfo is None:
        # Assume stored timestamps are UTC if naive.
        dt = UTC.localize(dt)
    else:
        dt = dt.astimezone(UTC)
    return dt


def next_trading_session_close(dt_utc: datetime) -> datetime:
    """
    If dt is during open market hours, return that session close.
    If dt is outside market hours/weekend/holiday, return next session close.
    Returned as timezone-aware UTC datetime.
    """
    dt_utc = dt_utc.astimezone(UTC)
    dt_ny = dt_utc.astimezone(NY)

    # Build a session window
    start = (dt_ny.date() - timedelta(days=10)).isoformat()
    end = (dt_ny.date() + timedelta(days=30)).isoformat()
    sessions = CAL.sessions_in_range(start, end)

    # Find first session label with date >= dt date
    session_label = None
    for s in sessions:
        if s.date() >= dt_ny.date():
            session_label = s
            break
    if session_label is None:
        session_label = sessions[-1]

    open_utc = CAL.session_open(session_label)
    close_utc = CAL.session_close(session_label)

    if open_utc <= pd.Timestamp(dt_utc) <= close_utc:
        return close_utc.to_pydatetime().replace(tzinfo=UTC)

    # outside current session => next session close
    # If dt is before open on a session day, use that session close.
    if dt_ny.date() == session_label.date() and pd.Timestamp(dt_utc) < open_utc:
        return close_utc.to_pydatetime().replace(tzinfo=UTC)

    next_label = CAL.next_session_label(session_label)
    next_close_utc = CAL.session_close(next_label)
    return next_close_utc.to_pydatetime().replace(tzinfo=UTC)


def add_horizon_then_snap_close(entry_time_utc: datetime, horizon: Literal["1d", "7d"], mode: Literal["calendar", "trading"]) -> datetime:
    """
    Compute an exit time and snap it to the appropriate session close.
    - calendar: exit = entry + N calendar days, then snap to next trading session close
    - trading : exit = close of (entry snapped) + N trading sessions, take that session close
    """
    entry_time_utc = entry_time_utc.astimezone(UTC)

    if mode == "calendar":
        raw_exit = entry_time_utc + (timedelta(days=1) if horizon == "1d" else timedelta(days=7))
        return next_trading_session_close(raw_exit)

    # trading mode
    base_close = next_trading_session_close(entry_time_utc)
    base_ny_date = base_close.astimezone(NY).date()

    start = (base_ny_date - timedelta(days=10)).isoformat()
    end = (base_ny_date + timedelta(days=90)).isoformat()
    sessions = list(CAL.sessions_in_range(start, end))

    base_idx = None
    for i, s in enumerate(sessions):
        if s.date() == base_ny_date:
            base_idx = i
            break
    if base_idx is None:
        # fallback to first session >= base date
        for i, s in enumerate(sessions):
            if s.date() >= base_ny_date:
                base_idx = i
                break
    if base_idx is None:
        raise RuntimeError("Could not locate base trading session.")

    n = 1 if horizon == "1d" else 7
    target_label = sessions[base_idx + n]
    target_close_utc = CAL.session_close(target_label)
    return target_close_utc.to_pydatetime().replace(tzinfo=UTC)


def to_yf_date(dt_utc: datetime) -> date:
    """
    Convert UTC datetime to exchange-local date for daily bars.
    We use the date in America/New_York.
    """
    return dt_utc.astimezone(NY).date()


# ----------------------------
# Helpers: price fetching (yfinance daily close)
# ----------------------------
_price_cache: Dict[Tuple[str, date], float] = {}

def get_daily_close(ticker: str, d: date) -> Optional[float]:
    """
    Fetch daily close price for ticker on date d (NY exchange date).
    Uses a small cache to reduce repeated calls.
    """
    key = (ticker.upper(), d)
    if key in _price_cache:
        return _price_cache[key]

    # yfinance: use an inclusive window around the date
    # We'll request [d-2, d+2] and pick the exact row if exists.
    start = (pd.Timestamp(d) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(d) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        # yfinance index is timezone-naive but represents exchange dates
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        row = df[df["Date"] == d]
        if row.empty:
            return None
        close = float(row.iloc[0]["Close"])
        _price_cache[key] = close
        return close
    except Exception:
        return None


# ----------------------------
# Supabase IO
# ----------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@dataclass
class ActionRow:
    id: str
    ticker: str
    action: str
    entry_time: datetime  # UTC


def fetch_actions_needing_evals(limit: int = 200) -> List[ActionRow]:
    """
    Pull recent actions. Since we can't know which horizons already exist without joins,
    we fetch recent actions and decide per horizon using a lookup table of existing evals.
    """
    # Adjust ordering/filters to your schema. If you have entry_time column, use it.
    resp = (
        supabase.table(ACTIONS_TABLE)
        .select("id,ticker,action,entry_time,created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    rows = resp.data or []
    out: List[ActionRow] = []
    for r in rows:
        entry_raw = r.get("entry_time") or r.get("created_at")
        if not entry_raw:
            continue
        out.append(
            ActionRow(
                id=str(r["id"]),
                ticker=str(r["ticker"]).upper(),
                action=str(r.get("action") or r.get("side") or "hold").lower(),
                entry_time=parse_dt(entry_raw),
            )
        )
    return out


def fetch_existing_eval_keys(action_ids: List[str]) -> set[Tuple[str, str]]:
    """
    Return a set of (action_id, horizon) that already exist.
    Requires agent_evals has a horizon column.
    """
    if not action_ids:
        return set()

    # Supabase "in" filter expects list
    resp = (
        supabase.table(EVALS_TABLE)
        .select("action_id,horizon")
        .in_("action_id", action_ids)
        .execute()
    )
    existing = set()
    for r in (resp.data or []):
        if r.get("action_id") and r.get("horizon"):
            existing.add((str(r["action_id"]), str(r["horizon"])))
    return existing


def upsert_eval(row: Dict[str, Any]) -> None:
    """
    Upsert by unique constraint on (action_id, horizon).
    """
    # If you don't have an upsert constraint yet, create it:
    #   create unique index agent_evals_action_horizon_uq on agent_evals(action_id, horizon);
    supabase.table(EVALS_TABLE).upsert(row, on_conflict="action_id,horizon").execute()


# ----------------------------
# Return logic
# ----------------------------
def compute_return(entry_px: float, exit_px: float, action: str) -> float:
    """
    Defines returns:
    - buy/long: (exit/entry - 1)
    - sell/short: -(exit/entry - 1)
    - hold: 0 (or treat as long/neutral; choose your preference)
    """
    base = (exit_px / entry_px) - 1.0
    if action in ("sell", "short"):
        return -base
    if action in ("hold", "neutral"):
        return 0.0
    return base  # buy/long default


# ----------------------------
# Main
# ----------------------------
def main():
    actions = fetch_actions_needing_evals(limit=300)
    if not actions:
        print("No actions found.")
        return

    existing = fetch_existing_eval_keys([a.id for a in actions])

    created = 0
    skipped = 0
    skip_reasons: Dict[str, int] = {}

    for a in actions:
        for horizon in HORIZONS:
            key = (a.id, horizon)
            if key in existing:
                skipped += 1
                skip_reasons["already_has_eval"] = skip_reasons.get("already_has_eval", 0) + 1
                continue

            # Compute snapped exit close time
            exit_time_utc = add_horizon_then_snap_close(a.entry_time, horizon=horizon, mode=HORIZON_MODE)

            # Daily closes (entry uses its exchange date close; same for exit)
            entry_date = to_yf_date(a.entry_time)
            exit_date = to_yf_date(exit_time_utc)

            entry_px = get_daily_close(a.ticker, entry_date)
            exit_px = get_daily_close(a.ticker, exit_date)

            if entry_px is None or exit_px is None:
                skipped += 1
                skip_reasons["missing_price"] = skip_reasons.get("missing_price", 0) + 1
                continue

            ret = compute_return(entry_px, exit_px, a.action)

            payload = {
                "action_id": a.id,
                "ticker": a.ticker,
                "horizon": horizon,
                "entry_time": a.entry_time.isoformat(),
                "exit_time": exit_time_utc.isoformat(),
                "entry_price": entry_px,
                "exit_price": exit_px,
                "return": ret,
                # Optional debug fields
                "entry_date": entry_date.isoformat(),
                "exit_date": exit_date.isoformat(),
                "created_at": datetime.now(tz=UTC).isoformat(),
            }

            try:
                upsert_eval(payload)
                created += 1
            except Exception as e:
                skipped += 1
                skip_reasons["upsert_failed"] = skip_reasons.get("upsert_failed", 0) + 1
                print(f"Upsert failed for action_id={a.id}, horizon={horizon}: {e}", file=sys.stderr)

    print(f"evaluate_once: total_actions_checked={len(actions) * len(HORIZONS)} created_evals={created} skipped={skipped}")
    if skip_reasons:
        print(f"Skip reasons: {skip_reasons}")


if __name__ == "__main__":
    main()
