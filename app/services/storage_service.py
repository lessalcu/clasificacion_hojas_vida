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
            file_bytes = file_obj.read()

        result = bucket.upload(
            path=remote_path,
            file=file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": str(upsert).lower(),
            },
        )

        return result

    def upload_fileobj(
        self,
        bucket_name: str,
        remote_path: str,
        file_obj,
        content_type: str,
        upsert: bool = False,
    ):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)

        if hasattr(file_obj, "seek"):
            file_obj.seek(0)

        if not hasattr(file_obj, "read"):
            raise TypeError("file_obj must be a readable file-like object")

        file_bytes = file_obj.read()

        result = bucket.upload(
            path=remote_path,
            file=file_bytes,
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
    
    def upload_bytes(
        self,
        bucket_name: str,
        remote_path: str,
        file_bytes: bytes,
        content_type: str,
        upsert: bool = False,
    ):
        supabase = get_supabase()
        return supabase.storage.from_(bucket_name).upload(
            path=remote_path,
            file=file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": str(upsert).lower(),
            },
        )

    def download_file(self, bucket_name: str, remote_path: str) -> bytes:
        supabase = get_supabase()
        return supabase.storage.from_(bucket_name).download(remote_path)