import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.services.storage_service import StorageService

storage = StorageService()

bucket_name = "cv-raw"
candidate_id = "smoke-test"
remote_path = f"{candidate_id}/cv_test.pdf"
local_file_path = "tests/fixtures/cv_test.pdf"

print("1. UPLOAD")
upload_result = storage.upload_file(
    bucket_name=bucket_name,
    remote_path=remote_path,
    local_file_path=local_file_path,
    content_type="application/pdf",
    upsert=True,
)
print(upload_result)

print("2. LIST")
files = storage.list_files(bucket_name=bucket_name, folder=candidate_id)
print(files)

print("3. REMOVE")
deleted = storage.remove_files(bucket_name, [remote_path])
print(deleted)

print("STORAGE OK")