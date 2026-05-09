import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from app.services.job_profile_service import JobProfileService

service = JobProfileService()

payload = {
    "title": "Desarrollador Full Stack",
    "description": "Perfil objetivo para pruebas",
    "required_skills": ["python", "react"],
    "technologies": ["flask", "postgres", "supabase"],
    "languages": ["es", "en"]
}

print("1. INSERT")
created = service.create_job_profile(payload)
print(created)

if not created:
    raise Exception("Insert failed")

created_id = created["id"]

print("2. SELECT BY ID")
selected = service.get_job_profile_by_id(created_id)
print(selected)

print("3. UPDATE")
updated = service.update_job_profile(created_id, {
    "description": "Perfil objetivo actualizado desde smoke test"
})
print(updated)

print("4. SELECT ALL")
rows = service.get_job_profiles(limit=5)
print(rows)

print("5. DELETE")
deleted = service.delete_job_profile(created_id)
print(deleted)

print("CRUD OK")