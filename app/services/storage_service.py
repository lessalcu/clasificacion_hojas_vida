from app.services.supabase_client import get_supabase


class StorageService:
    def upload_file(
        self,
        bucket_name: str,
        remote_path: str,
        local_file_path: str,
        content_type: str,
        upsert: bool = False,
    ):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)

        with open(local_file_path, "rb") as file_obj:
            result = bucket.upload(
                path=remote_path,
                file=file_obj,
                file_options={
                    "content-type": content_type,
                    "upsert": str(upsert).lower(),
                },
            )

        return result

    def list_files(self, bucket_name: str, folder: str = ""):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)
        return bucket.list(folder)

    def remove_files(self, bucket_name: str, remote_paths: list[str]):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)
        return bucket.remove(remote_paths)