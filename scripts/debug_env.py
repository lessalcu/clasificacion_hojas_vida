import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
print("BASE_DIR:", BASE_DIR)

env_path = BASE_DIR / ".env"
print("ENV PATH:", env_path)
print("ENV EXISTS:", env_path.exists())

load_dotenv(env_path)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SECRET_KEY")

print("URL:", repr(url))
print("KEY EXISTS:", bool(key))