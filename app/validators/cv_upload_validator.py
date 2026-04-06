from app.errors.exceptions import ValidationError

ALLOWED_EXTENSIONS = {"pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}


def get_extension(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def validate_pdf_file(file_storage, max_size_bytes: int):
    if file_storage is None:
        raise ValidationError("File is required")

    filename = file_storage.filename or ""
    if not filename.strip():
        raise ValidationError("Filename is required")

    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError("Only PDF files are allowed")

    mime_type = file_storage.mimetype or ""
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValidationError("Invalid MIME type. Expected application/pdf")

    file_storage.stream.seek(0, 2)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size <= 0:
        raise ValidationError("File is empty")

    if size > max_size_bytes:
        raise ValidationError(
            f"File exceeds the maximum allowed size of {max_size_bytes} bytes"
        )

    return {
        "original_filename": filename,
        "file_extension": extension,
        "mime_type": mime_type,
        "file_size_bytes": size,
    }