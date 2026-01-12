#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from datetime import datetime, time, timedelta

import pytz
import exchange_calendars as xcals


NY = pytz.timezone("America/New_York")
CAL = xcals.get_calendar("XNYS")


def is_trading_session_day(d) -> bool:
    return bool(CAL.is_session(d))


def in_market_close_window(now_ny: datetime) -> bool:
    """
    Run only in a tight window shortly after 4:00pm ET.
    Allows buffer for data availability and DST scheduling.
    """
    start = time(16, 0)   # 4:00pm ET
    end = time(16, 20)    # 4:20pm ET
    t = now_ny.time().replace(tzinfo=None)
    return start <= t <= end


def run_script(script_name: str) -> int:
    print(f"Running {script_name} ...")
    r = subprocess.run([os.sys.executable, script_name], check=False)
    return r.returncode


def main():
    now_ny = datetime.now(tz=NY)
    today = now_ny.date()

    force = os.getenv("FORCE_RUN", "").strip() == "1"

    # 1) Must be a trading session (unless forced)
    if not force and not is_trading_session_day(today):
        print(f"[SKIP] {today} is not a trading session day.")
        print("       Set FORCE_RUN=1 to bypass this check (useful for manual tests).")
        return
    
    if force and not is_trading_session_day(today):
        print(f"[FORCE] Bypassing trading-session check. today_ny={today}")


    # 2) Must be in close window (unless forced)
    force = os.getenv("FORCE_RUN", "").strip() == "1"
    if not force and not in_market_close_window(now_ny):
        print(f"[SKIP] Not in market-close window. now_ny={now_ny.isoformat()}")
        print("       Set FORCE_RUN=1 to bypass this check (useful for manual tests).")
        return
    
    if force:
        print(f"[FORCE] Bypassing close-window check. now_ny={now_ny.isoformat()}")


    # 3) Run your pipeline
    # Generate actions at/after close
    rc1 = run_script("generate_actions.py")
    if rc1 != 0:
        print(f"[WARN] generate_actions.py exited with code {rc1}")

    # Evaluate actions (creates 1d/7d evals)
    rc2 = run_script("evaluate_actions.py")
    if rc2 != 0:
        print(f"[WARN] evaluate_actions.py exited with code {rc2}")

    print("[OK] Market-close pipeline completed.")


if __name__ == "__main__":
    main()
