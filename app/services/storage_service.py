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

        stream = file_obj.stream if hasattr(file_obj, "stream") else file_obj

        if hasattr(stream, "seek"):
            stream.seek(0)

        payload = stream.read() if hasattr(stream, "read") else stream

        result = bucket.upload(
            path=remote_path,
            file=payload,
            file_options={
                "content-type": content_type,
                "upsert": str(upsert).lower(),
            },
        )

        return result
    
    def upload_json(
        self,
        bucket_name: str,
        remote_path: str,
        payload: dict,
        upsert: bool = True,
    ):
        import io
        import json

        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        file_obj = io.BytesIO(json_bytes)

        return self.upload_fileobj(
            bucket_name=bucket_name,
            remote_path=remote_path,
            file_obj=file_obj,
            content_type="application/json",
            upsert=upsert,
        )

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
        data: bytes | None = None,
        content_type: str = "application/octet-stream",
        upsert: bool = False,
        file_bytes: bytes | None = None,
    ):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)

        payload = data if data is not None else file_bytes
        if payload is None:
            raise ValueError("upload_bytes requires 'data' or 'file_bytes'")

        result = bucket.upload(
            path=remote_path,
            file=payload,
            file_options={
                "content-type": content_type,
                "upsert": str(upsert).lower(),
            },
        )

        return result

    def download_file(self, bucket_name: str, remote_path: str):
        supabase = get_supabase()
        bucket = supabase.storage.from_(bucket_name)
        return bucket.download(remote_path)