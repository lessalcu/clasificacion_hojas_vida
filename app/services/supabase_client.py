import os
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase

    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_KEY")

        if not url:
            raise RuntimeError("Missing SUPABASE_URL in environment variables")

        if not key:
            raise RuntimeError("Missing SUPABASE_SECRET_KEY in environment variables")

        _supabase = create_client(url, key)

    return _supabase