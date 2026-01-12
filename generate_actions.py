#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
from dotenv import load_dotenv
from supabase import create_client

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TICKERS_TABLE = os.getenv("TICKERS_TABLE", "tickers")
ACTIONS_TABLE = os.getenv("ACTIONS_TABLE", "agent_actions")

# If you don't want to create a tickers table yet, set WATCHLIST="AAPL,MSFT,NVDA"
WATCHLIST = os.getenv("WATCHLIST", "").strip()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheap + reliable
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "50"))

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC


# ----------------------------
# Helpers
# ----------------------------
def must_env():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


def ny_run_date() -> str:
    # run_date in NY (market date)
    return datetime.now(tz=NY).date().isoformat()


def fetch_watchlist(supabase) -> List[str]:
    # Option 1: env var
    if WATCHLIST:
        return [t.strip().upper() for t in WATCHLIST.split(",") if t.strip()][:MAX_TICKERS]

    # Option 2: table
    resp = (
        supabase.table(TICKERS_TABLE)
        .select("ticker")
        .eq("is_active", True)
        .limit(MAX_TICKERS)
        .execute()
    )
    rows = resp.data or []
    return [str(r["ticker"]).upper() for r in rows if r.get("ticker")]


def already_generated_today(supabase, run_date: str, ticker: str) -> bool:
    # relies on unique index (run_date, ticker) but we can check before insert for clean logs
    resp = (
        supabase.table(ACTIONS_TABLE)
        .select("id")
        .eq("run_date", run_date)
        .eq("ticker", ticker)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def build_prompt(ticker: str) -> str:
    # Keep it simple and deterministic. Later you can add signals, news, fundamentals, etc.
    return f"""
You are a trading decision agent for US equities. You must output a single JSON object.

Goal: Decide a 1-week directional action for {ticker} as of market close today.

Rules:
- Action must be one of: "buy", "sell", "hold"
- confidence must be a number between 0 and 1
- reason: concise (<= 60 words), focus on generalizable signals (trend, volatility, risk) without claiming you saw live prices/news unless provided
- Do NOT include any extra keys. Output JSON only.

Example:
{{"action":"hold","confidence":0.55,"reason":"..."}}""".strip()


def call_agent(client: OpenAI, prompt: str) -> Dict[str, Any]:
    # Use JSON mode to reduce parsing issues
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    data = json.loads(content)

    # Validate
    action = str(data.get("action", "")).lower().strip()
    if action not in ("buy", "sell", "hold"):
        raise ValueError(f"Invalid action: {action}")

    conf = float(data.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))

    reason = str(data.get("reason", "")).strip()
    if len(reason) > 600:
        reason = reason[:600]

    return {"action": action, "confidence": conf, "reason": reason}


def upsert_action(
    supabase,
    run_date: str,
    ticker: str,
    agent_out: Dict[str, Any],
    prompt: str,
):
    payload = {
        "run_date": run_date,
        "ticker": ticker,
        "action": agent_out["action"],
        "confidence": agent_out["confidence"],
        "reason": agent_out["reason"],
        "prompt": prompt,
        "model": MODEL,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }

    # Upsert by unique index on (run_date, ticker)
    supabase.table(ACTIONS_TABLE).upsert(payload, on_conflict="run_date,ticker").execute()


# ----------------------------
# Main
# ----------------------------
def main():
    must_env()

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)

    run_date = ny_run_date()
    tickers = fetch_watchlist(supabase)

    if not tickers:
        print(f"No tickers found. Set WATCHLIST or create table '{TICKERS_TABLE}'.")
        return

    created = 0
    skipped = 0
    failed = 0

    print(f"Generating actions for run_date={run_date}, tickers={len(tickers)}")

    for t in tickers:
        try:
            if already_generated_today(supabase, run_date, t):
                skipped += 1
                print(f"[SKIP] {t} already has action for {run_date}")
                continue

            prompt = build_prompt(t)
            out = call_agent(client, prompt)
            upsert_action(supabase, run_date, t, out, prompt)

            created += 1
            print(f"[OK] {t} -> {out['action']} (conf={out['confidence']:.2f})")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t}: {e}")

    print(f"Done. created={created} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
