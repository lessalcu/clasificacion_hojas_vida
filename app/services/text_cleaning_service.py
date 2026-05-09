import re
import unicodedata


class TextCleaningService:
    def clean(self, text: str | None) -> str:
        if not text:
            return ""

        cleaned = str(text)

        cleaned = unicodedata.normalize("NFKC", cleaned)
        cleaned = cleaned.replace("\x00", " ")

        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[\u200b\u200c\u200d]", "", cleaned)

        cleaned = re.sub(r"(?i)página\s+\d+\s+de\s+\d+", " ", cleaned)
        cleaned = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", " ", cleaned)

        cleaned = cleaned.strip()

        return cleaned

    def is_valid_for_training(self, text: str | None, min_length: int = 80) -> bool:
        cleaned = self.clean(text)
        return len(cleaned) >= min_length
