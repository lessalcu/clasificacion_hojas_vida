import json
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import pdfplumber
import pymupdf

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.storage_service import StorageService


class PdfTextExtractionService:
    SECTION_HEADERS = [
        (r"\bDatos de contacto\b", "DATOS DE CONTACTO"),
        (r"\bExperiencia Laboral\b", "EXPERIENCIA LABORAL"),
        (r"\bEstudios\b", "ESTUDIOS"),
        (r"\bReferencias\b", "REFERENCIAS"),
        (r"\bObjetivo laboral\b", "OBJETIVO LABORAL"),
        (r"\bConocimientos y habilidades\b", "CONOCIMIENTOS Y HABILIDADES"),
        (r"\bOtros conocimientos\b", "OTROS CONOCIMIENTOS"),
        (r"\bIdiomas\b", "IDIOMAS"),
    ]

    EXPLODED_LINE_RE = re.compile(
        r"^(?:[A-Za-zÁÉÍÓÚáéíóúÑñ0-9#:/().,-]\s+){4,}[A-Za-zÁÉÍÓÚáéíóúÑñ0-9#:/().,-]+$"
    )
    SINGLE_CHAR_RE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]$")
    VISIBLE_CHARS_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]")

    def __init__(self):
        self.storage_service = StorageService()
        self.candidate_source_repository = CandidateSourceRepository()

    def extract_from_source(self, source_id: str):
        source = self.candidate_source_repository.get_by_id(source_id)

        if not source:
            raise NotFoundError("Candidate source not found")

        if source["source_type"] != "pdf":
            raise ValidationError("Only PDF sources can be processed")

        if not source.get("storage_bucket") or not source.get("storage_path"):
            raise ValidationError("Source does not have a valid storage location")

        pdf_bytes = self.storage_service.download_file(
            bucket_name=source["storage_bucket"],
            remote_path=source["storage_path"],
        )

        raw_text, page_count, extractor_used, quality_score = self._extract_native_text(
            pdf_bytes
        )
        cleaned_text = self._clean_text(raw_text)

        if not self._has_useful_text(cleaned_text):
            updated = self.candidate_source_repository.update(source_id, {
                "extraction_status": "failed",
                "extraction_error": "PDF has no usable native text",
            })
            return {
                "status": "failed",
                "candidate_source": updated,
                "reason": "PDF has no usable native text",
            }

        normalized_path = f"{source['candidate_id']}/{source['id']}/v1.json"

        artifact = {
            "candidate_id": source["candidate_id"],
            "candidate_source_id": source["id"],
            "source_type": source["source_type"],
            "original_filename": source["original_filename"],
            "page_count": page_count,
            "text_length": len(cleaned_text),
            "ocr_used": False,
            "extractor": extractor_used,
            "quality_score": quality_score,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "text": cleaned_text,
        }

        self.storage_service.upload_bytes(
            bucket_name="cv-normalized",
            remote_path=normalized_path,
            file_bytes=json.dumps(artifact, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            upsert=True,
        )

        updated = self.candidate_source_repository.update(source_id, {
            "extraction_status": "processed",
            "normalized_bucket": "cv-normalized",
            "normalized_path": normalized_path,
            "extracted_text_length": len(cleaned_text),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extraction_error": None,
        })

        return {
            "status": "processed",
            "candidate_source": updated,
            "artifact": {
                "bucket": "cv-normalized",
                "path": normalized_path,
                "page_count": page_count,
                "text_length": len(cleaned_text),
                "extractor": extractor_used,
                "quality_score": quality_score,
            },
        }

    def _extract_native_text(self, pdf_bytes: bytes):
        candidates: list[dict[str, Any]] = []

        pdfplumber_result = self._extract_with_pdfplumber(pdf_bytes)
        if pdfplumber_result["text"].strip():
            candidates.append(pdfplumber_result)

        pymupdf_result = self._extract_with_pymupdf(pdf_bytes)
        if pymupdf_result["text"].strip():
            candidates.append(pymupdf_result)

        if not candidates:
            return "", 0, "none", 0

        best = max(candidates, key=lambda item: item["quality_score"])
        return (
            best["text"],
            best["page_count"],
            best["extractor"],
            best["quality_score"],
        )

    def _extract_with_pdfplumber(self, pdf_bytes: bytes):
        page_texts = []
        page_count = 0

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)

            for page in pdf.pages:
                words = page.extract_words(
                    x_tolerance=2,
                    x_tolerance_ratio=0.35,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                    split_at_punctuation=False,
                    expand_ligatures=True,
                )

                page_texts.append(self._reconstruct_text_from_pdfplumber_words(words))

        text = "\n\n".join(page_texts).strip()
        return {
            "extractor": "pdfplumber",
            "text": text,
            "page_count": page_count,
            "quality_score": self._score_text_quality(text),
        }

    def _extract_with_pymupdf(self, pdf_bytes: bytes):
        page_texts = []
        page_count = 0

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            page_count = len(doc)

            for page in doc:
                words = page.get_text("words", delimiters=None)
                page_texts.append(self._reconstruct_text_from_pymupdf_words(words))
        finally:
            doc.close()

        text = "\n\n".join(page_texts).strip()
        return {
            "extractor": "pymupdf",
            "text": text,
            "page_count": page_count,
            "quality_score": self._score_text_quality(text),
        }

    def _reconstruct_text_from_pdfplumber_words(self, words: list[dict[str, Any]]) -> str:
        if not words:
            return ""

        sorted_words = sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"])))

        lines: list[list[dict[str, Any]]] = []
        current_line: list[dict[str, Any]] = []
        current_top: float | None = None
        line_tolerance = 3.0

        for word in sorted_words:
            top = float(word["top"])
            if current_top is None or abs(top - current_top) <= line_tolerance:
                current_line.append(word)
                current_top = top if current_top is None else min(current_top, top)
            else:
                lines.append(sorted(current_line, key=lambda w: float(w["x0"])))
                current_line = [word]
                current_top = top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: float(w["x0"])))

        text_lines = [" ".join(w["text"] for w in line).strip() for line in lines]
        return "\n".join(line for line in text_lines if line)

    def _reconstruct_text_from_pymupdf_words(self, words: list[tuple]) -> str:
        if not words:
            return ""

        # PyMuPDF words: x0, y0, x1, y1, text, block_no, line_no, word_no
        sorted_words = sorted(words, key=lambda w: (w[5], w[6], w[0]))

        lines: dict[tuple[int, int], list[tuple]] = {}
        for word in sorted_words:
            key = (int(word[5]), int(word[6]))
            lines.setdefault(key, []).append(word)

        ordered_keys = sorted(lines.keys(), key=lambda k: (k[0], k[1]))
        text_lines = []

        for key in ordered_keys:
            line_words = sorted(lines[key], key=lambda w: w[0])
            text_lines.append(" ".join(w[4] for w in line_words).strip())

        return "\n".join(line for line in text_lines if line)

    def _score_text_quality(self, text: str) -> int:
        if not text.strip():
            return 0

        visible_chars = len(self.VISIBLE_CHARS_RE.findall(text))
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        exploded_lines = sum(
            1 for line in non_empty_lines if self.EXPLODED_LINE_RE.fullmatch(line)
        )

        section_hits = 0
        lowered = text.lower()
        for pattern, _label in self.SECTION_HEADERS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                section_hits += 1

        # penaliza fuertemente líneas explotadas tipo "D e s a r r o l l a d o r"
        score = visible_chars + (section_hits * 100) - (exploded_lines * 250)

        # bonus por texto suficientemente largo
        if len(text) >= 500:
            score += 200

        return score

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\uf0b7", "•")
        text = re.sub(r"\r\n?", "\n", text)

        text = self._apply_known_replacements(text)

        # separa letras y números pegados
        text = re.sub(r"([A-Za-zÁÉÍÓÚáéíóúÑñ])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Za-zÁÉÍÓÚáéíóúÑñ])", r"\1 \2", text)

        # normaliza algunos guiones/slashes
        text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)
        text = re.sub(r"\s*/\s*", " / ", text)

        # normaliza espacios sin destruir líneas
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)

        # reetiqueta encabezados conocidos
        for pattern, label in self.SECTION_HEADERS:
            text = re.sub(pattern, f"\n{label}\n", text, flags=re.IGNORECASE)

        lines = [re.sub(r"\s{2,}", " ", line).strip() for line in text.split("\n")]

        normalized_lines = []
        for line in lines:
            if line and self.EXPLODED_LINE_RE.fullmatch(line):
                normalized_lines.append(self._collapse_exploded_line(line))
            else:
                normalized_lines.append(line)

        compacted_lines = []
        previous_blank = False
        for line in normalized_lines:
            if not line:
                if not previous_blank:
                    compacted_lines.append("")
                previous_blank = True
                continue

            compacted_lines.append(line)
            previous_blank = False

        text = "\n".join(compacted_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _collapse_exploded_line(self, line: str) -> str:
        tokens = line.split()
        if not tokens:
            return line

        rebuilt = []
        buffer = []

        def flush_buffer():
            nonlocal buffer
            if buffer:
                rebuilt.append("".join(buffer))
                buffer = []

        for token in tokens:
            if self.SINGLE_CHAR_RE.fullmatch(token):
                buffer.append(token)
            else:
                flush_buffer()
                rebuilt.append(token)

        flush_buffer()

        collapsed = " ".join(rebuilt)
        collapsed = self._apply_known_replacements(collapsed)

        # algunos nombres frecuentes
        collapsed = re.sub(r"\bReactNative\b", "React Native", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"\bPowerBI\b", "Power BI", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"\bSqlServer\b", "Sql Server", collapsed, flags=re.IGNORECASE)
        collapsed = re.sub(r"\bCSharp\b", "C Sharp", collapsed, flags=re.IGNORECASE)

        return collapsed

    def _apply_known_replacements(self, text: str) -> str:
        replacements = {
            r"\bAlpresente\b": "Al presente",
            r"\bEncurso\b": "En curso",
            r"\bOtrosconocimientos\b": "Otros conocimientos",
            r"\bDatosdecontacto\b": "Datos de contacto",
            r"\bObjetivolaboral\b": "Objetivo laboral",
            r"\bExperienciaLaboral\b": "Experiencia Laboral",
            r"\bConocimientosyhabilidades\b": "Conocimientos y habilidades",
            r"\bReferenciaacademica\b": "Referencia académica",
            r"\bmeybili200219@gmail\.com\b": "meybili 200219@gmail.com",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _has_useful_text(self, text: str) -> bool:
        if not text:
            return False

        visible_chars = len(self.VISIBLE_CHARS_RE.findall(text))
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        exploded_lines = sum(
            1 for line in non_empty_lines if self.EXPLODED_LINE_RE.fullmatch(line)
        )

        if len(text) < 120 or visible_chars < 40:
            return False

        if non_empty_lines and (exploded_lines / len(non_empty_lines)) > 0.4:
            return False

        return True