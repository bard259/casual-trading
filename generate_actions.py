# generate_actions.py
from __future__ import annotations
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
UTC = pytz.UTC

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
ACTIONS_TABLE = os.getenv("ACTIONS_TABLE", "agent_actions")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def main():
    # TODO: Replace this with your real agent generation logic
    # Example dummy action
    payload = {
        "ticker": "SPY",
        "action": "hold",
        "entry_time": datetime.now(tz=UTC).isoformat(),
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    supabase.table(ACTIONS_TABLE).insert(payload).execute()
    print("Inserted dummy action.")

if __name__ == "__main__":
    main()
