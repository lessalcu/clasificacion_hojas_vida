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
        (r"\bContacto\b", "CONTACTO"),
        (r"\bExperiencia Laboral\b", "EXPERIENCIA LABORAL"),
        (r"\bExperiencia Profesional\b", "EXPERIENCIA PROFESIONAL"),
        (r"\bEstudios\b", "ESTUDIOS"),
        (r"\bFormación Académica\b", "FORMACIÓN ACADÉMICA"),
        (r"\bFormacion Academica\b", "FORMACIÓN ACADÉMICA"),
        (r"\bReferencias\b", "REFERENCIAS"),
        (r"\bObjetivo laboral\b", "OBJETIVO LABORAL"),
        (r"\bAcerca de mí\b", "ACERCA DE MÍ"),
        (r"\bAcerca de mi\b", "ACERCA DE MÍ"),
        (r"\bConocimientos y habilidades\b", "CONOCIMIENTOS Y HABILIDADES"),
        (r"\bHabilidades\b", "HABILIDADES"),
        (r"\bOtros conocimientos\b", "OTROS CONOCIMIENTOS"),
        (r"\bOtros cursos\b", "OTROS CURSOS"),
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

        extracted = self._extract_native_text(pdf_bytes)
        raw_text = extracted["text"]
        blocks = extracted["blocks"]
        page_count = extracted["page_count"]
        extractor_used = extracted["extractor"]
        quality_score = extracted["quality_score"]

        cleaned_text = self._clean_text(raw_text)
        cleaned_blocks = [
            self._clean_block(block)
            for block in blocks
            if block.get("text", "").strip()
        ]

        if not self._has_useful_text(cleaned_text):
            updated = self.candidate_source_repository.update(
                source_id,
                {
                    "extraction_status": "failed",
                    "extraction_error": "PDF has no usable native text",
                },
            )
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
            "blocks": cleaned_blocks,
        }

        self.storage_service.upload_bytes(
            bucket_name="cv-normalized",
            remote_path=normalized_path,
            file_bytes=json.dumps(artifact, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            upsert=True,
        )

        updated = self.candidate_source_repository.update(
            source_id,
            {
                "extraction_status": "processed",
                "normalized_bucket": "cv-normalized",
                "normalized_path": normalized_path,
                "extracted_text_length": len(cleaned_text),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
                "extraction_error": None,
            },
        )

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
                "block_count": len(cleaned_blocks),
            },
        }

    def _extract_native_text(self, pdf_bytes: bytes):
        candidates: list[dict[str, Any]] = []

        try:
            pdfplumber_result = self._extract_with_pdfplumber(pdf_bytes)
            if pdfplumber_result["text"].strip():
                candidates.append(pdfplumber_result)
        except Exception as exc:
            print("DEBUG pdfplumber ERROR:", type(exc).__name__, str(exc))

        try:
            pymupdf_result = self._extract_with_pymupdf(pdf_bytes)
            if pymupdf_result["text"].strip():
                candidates.append(pymupdf_result)
        except Exception as exc:
            print("DEBUG pymupdf ERROR:", type(exc).__name__, str(exc))

        if not candidates:
            return {
                "text": "",
                "blocks": [],
                "page_count": 0,
                "extractor": "none",
                "quality_score": 0,
            }

        return max(candidates, key=lambda item: item["quality_score"])

    def _extract_with_pdfplumber(self, pdf_bytes: bytes):
        all_blocks: list[dict[str, Any]] = []
        page_count = 0

        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)

            for page_index, page in enumerate(pdf.pages, start=1):
                page_width = float(page.width) if page.width is not None else 0.0

                words = page.extract_words(
                    x_tolerance=2,
                    x_tolerance_ratio=0.35,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                    split_at_punctuation=False,
                    expand_ligatures=True,
                )

                page_blocks = self._page_blocks_from_words(
                    words=words,
                    page_index=page_index,
                    page_width=page_width,
                )
                all_blocks.extend(page_blocks)

        text = self._compose_text_from_blocks(all_blocks)

        return {
            "extractor": "pdfplumber_blocks",
            "text": text,
            "blocks": all_blocks,
            "page_count": page_count,
            "quality_score": self._score_text_quality(text, all_blocks),
        }

    def _extract_with_pymupdf(self, pdf_bytes: bytes):
        all_blocks: list[dict[str, Any]] = []
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        try:
            for page_index, page in enumerate(doc, start=1):
                page_width = float(page.rect.width)

                raw_blocks = [
                    block for block in page.get_text("blocks")
                    if block[4] and block[4].strip()
                ]

                centers = sorted(
                    (float(block[0]) + float(block[2])) / 2.0
                    for block in raw_blocks
                )

                split_x = page_width / 2.0

                if len(centers) >= 4:
                    max_gap = 0.0
                    for i in range(len(centers) - 1):
                        gap = centers[i + 1] - centers[i]
                        if gap > max_gap:
                            max_gap = gap
                            split_x = (centers[i] + centers[i + 1]) / 2.0

                for block in raw_blocks:
                    x0, y0, x1, y1, text, *_rest = block
                    if not text or not text.strip():
                        continue

                    x0 = float(x0)
                    y0 = float(y0)
                    x1 = float(x1)
                    y1 = float(y1)

                    column = "left" if ((x0 + x1) / 2.0) < split_x else "right"

                    all_blocks.append(
                        {
                            "page": page_index,
                            "x0": round(x0, 2),
                            "top": round(y0, 2),
                            "x1": round(x1, 2),
                            "bottom": round(y1, 2),
                            "column": column,
                            "text": text.strip(),
                        }
                    )
        finally:
            doc.close()

        all_blocks = sorted(
            all_blocks,
            key=lambda b: (
                b["page"],
                0 if b["column"] == "left" else 1,
                b["top"],
                b["x0"],
            ),
        )

        text = self._compose_text_from_blocks(all_blocks)

        return {
            "extractor": "pymupdf_blocks",
            "text": text,
            "blocks": all_blocks,
            "page_count": len({b["page"] for b in all_blocks}),
            "quality_score": self._score_text_quality(text, all_blocks),
        }

    def _page_blocks_from_words(
    self,
    words: list[dict[str, Any]],
    page_index: int,
    page_width: float,
    ):
        if not words:
            return []

        page_width = float(page_width)
        split_x = self._estimate_column_split_x(words, page_width)

        left_words: list[dict[str, Any]] = []
        right_words: list[dict[str, Any]] = []

        for word in words:
            x0 = float(word["x0"])
            x1 = float(word["x1"])
            center_x = (x0 + x1) / 2.0

            if center_x < split_x:
                left_words.append(word)
            else:
                right_words.append(word)

        left_blocks = self._build_blocks_from_column_words(
            words=left_words,
            page_index=page_index,
            column="left",
        )
        right_blocks = self._build_blocks_from_column_words(
            words=right_words,
            page_index=page_index,
            column="right",
        )

        return sorted(
            left_blocks + right_blocks,
            key=lambda b: (
                b["page"],
                0 if b["column"] == "left" else 1,
                b["top"],
                b["x0"],
            ),
        )

    def _build_blocks_from_column_words(
        self,
        words: list[dict[str, Any]],
        page_index: int,
        column: str,
    ):
        if not words:
            return []

        sorted_words = sorted(
            words,
            key=lambda w: (round(float(w["top"]), 1), float(w["x0"])),
        )

        line_tolerance = 3.0
        lines: list[dict[str, Any]] = []
        current_line: list[dict[str, Any]] = []
        current_top = None

        for word in sorted_words:
            top = float(word["top"])

            if current_top is None or abs(top - current_top) <= line_tolerance:
                current_line.append(word)
                current_top = top if current_top is None else min(current_top, top)
            else:
                lines.append(self._build_line(current_line, column))
                current_line = [word]
                current_top = top

        if current_line:
            lines.append(self._build_line(current_line, column))

        lines = [line for line in lines if line["text"].strip()]
        lines = sorted(lines, key=lambda l: (l["top"], l["x0"]))

        blocks: list[dict[str, Any]] = []
        current_block = None

        for line in lines:
            if current_block is None:
                current_block = self._new_block_from_line(page_index, line)
                continue

            close_vertically = abs(line["top"] - current_block["bottom"]) <= 14
            similar_indent = abs(line["x0"] - current_block["x0"]) <= 80

            if close_vertically and similar_indent:
                current_block["text"] += "\n" + line["text"]
                current_block["x0"] = min(current_block["x0"], line["x0"])
                current_block["x1"] = max(current_block["x1"], line["x1"])
                current_block["bottom"] = line["bottom"]
            else:
                blocks.append(current_block)
                current_block = self._new_block_from_line(page_index, line)

        if current_block:
            blocks.append(current_block)

        return blocks

    def _build_line(self, words: list[dict[str, Any]], column: str):
        words = sorted(words, key=lambda w: float(w["x0"]))
        text = " ".join(w["text"] for w in words).strip()
        x0 = min(float(w["x0"]) for w in words)
        x1 = max(float(w["x1"]) for w in words)
        top = min(float(w["top"]) for w in words)
        bottom = max(float(w["bottom"]) for w in words)

        return {
            "text": text,
            "x0": round(x0, 2),
            "x1": round(x1, 2),
            "top": round(top, 2),
            "bottom": round(bottom, 2),
            "column": column,
        }

    def _new_block_from_line(self, page_index: int, line: dict[str, Any]):
        return {
            "page": page_index,
            "x0": line["x0"],
            "top": line["top"],
            "x1": line["x1"],
            "bottom": line["bottom"],
            "column": line["column"],
            "text": line["text"],
        }

    def _compose_text_from_blocks(self, blocks: list[dict[str, Any]]):
        ordered = sorted(
            blocks,
            key=lambda b: (
                b["page"],
                0 if b["column"] == "left" else 1,
                b["top"],
                b["x0"],
            ),
        )
        return "\n\n".join(
            block["text"].strip() for block in ordered if block["text"].strip()
        )

    def _score_text_quality(self, text: str, blocks: list[dict[str, Any]]):
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

        score = visible_chars + (section_hits * 120) - (exploded_lines * 250)

        if len(blocks) >= 4:
            score += 150

        if len(text) >= 500:
            score += 200

        return score

    def _clean_block(self, block: dict[str, Any]):
        cleaned = dict(block)
        cleaned["text"] = self._clean_text(block.get("text", ""))
        return cleaned

    def _clean_text(self, text: str):
        if not text:
            return ""

        text = text.replace("\x00", " ")
        text = text.replace("\uf0b7", "•")
        text = re.sub(r"\r\n?", "\n", text)

        text = self._apply_known_replacements(text)

        text = re.sub(r"([A-Za-zÁÉÍÓÚáéíóúÑñ])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([A-Za-zÁÉÍÓÚáéíóúÑñ])", r"\1 \2", text)
        text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)
        text = re.sub(r"\s*/\s*", " / ", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)

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
    def _estimate_column_split_x(self, words: list[dict[str, Any]], page_width: float) -> float:
        if not words:
            return float(page_width) / 2.0

        centers = sorted(
            (float(word["x0"]) + float(word["x1"])) / 2.0
            for word in words
        )

        if len(centers) < 6:
            return float(page_width) / 2.0

        max_gap = 0.0
        split_x = float(page_width) / 2.0

        for i in range(len(centers) - 1):
            left = centers[i]
            right = centers[i + 1]
            gap = right - left

            if gap > max_gap:
                max_gap = gap
                split_x = (left + right) / 2.0

        min_split = float(page_width) * 0.25
        max_split = float(page_width) * 0.75
        split_x = max(min_split, min(max_split, split_x))

        return split_x
    
    def _collapse_exploded_line(self, line: str):
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
        return self._apply_known_replacements(collapsed)

    def _apply_known_replacements(self, text: str):
        replacements = {
            r"\bAlpresente\b": "Al presente",
            r"\bEncurso\b": "En curso",
            r"\bOtrosconocimientos\b": "Otros conocimientos",
            r"\bDatosdecontacto\b": "Datos de contacto",
            r"\bObjetivolaboral\b": "Objetivo laboral",
            r"\bExperienciaLaboral\b": "Experiencia Laboral",
            r"\bConocimientosyhabilidades\b": "Conocimientos y habilidades",
            r"\bReferenciaacademica\b": "Referencia académica",
            r"\bReactNative\b": "React Native",
            r"\bPowerBI\b": "Power BI",
            r"\bSqlServer\b": "Sql Server",
            r"\bCSharp\b": "C Sharp",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _has_useful_text(self, text: str):
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