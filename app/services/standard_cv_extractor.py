import io
import re
from datetime import datetime, timezone
from typing import Any

import fitz
import pdfplumber


class StandardCvExtractor:
    def __init__(self, standard_cv_config: dict[str, Any]):
        self.config = standard_cv_config or {}

    def extract(self, pdf_bytes: bytes):
        pages = self._extract_pages(pdf_bytes)

        if not pages:
            return {
                "valid": False,
                "message": "PDF text could not be extracted",
            }

        page_count = len(pages)
        max_pages = self.config.get("max_pages")

        if max_pages and page_count > max_pages:
            return {
                "valid": False,
                "message": f"PDF exceeds the maximum supported pages for standard CV format: {max_pages}",
            }

        full_text = "\n\n".join(page["text"] for page in pages).strip()
        full_text = self._normalize_text(full_text)

        if not full_text:
            return {
                "valid": False,
                "message": "Extracted PDF text is empty",
            }

        split_result = self._split_standard_sections(full_text)
        sections = split_result["sections"]
        found_order = split_result["found_order"]

        validation = self._validate_standard_sections(
            sections=sections,
            found_order=found_order,
        )
        if not validation["valid"]:
            return validation

        section_blocks = self._build_section_blocks(sections)
        canonical_text = self._build_canonical_text(sections)
        quality_score = self._calculate_quality_score(sections, canonical_text)

        artifact_payload = {
            "page_count": page_count,
            "text_length": len(canonical_text),
            "ocr_used": False,
            "extractor": "standard_cv_template",
            "quality_score": quality_score,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "text": canonical_text,
            "sections": sections,
            "blocks": section_blocks,
            "standard_template_valid": True,
            "standard_template_version": "v1",
        }

        return {
            "valid": True,
            "message": "Standard CV extracted successfully",
            "artifact_payload": artifact_payload,
        }

    def _extract_pages(self, pdf_bytes: bytes):
        pages = []

        # Principal: pdfplumber
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(
                        x_tolerance=2,
                        y_tolerance=3,
                        layout=False,
                    ) or ""
                    text = self._normalize_text(text)

                    if text.strip():
                        pages.append(
                            {
                                "page": index,
                                "text": text,
                            }
                        )
        except Exception:
            pages = []

        # Fallback: PyMuPDF
        if not pages:
            try:
                document = fitz.open(stream=pdf_bytes, filetype="pdf")
                for index, page in enumerate(document, start=1):
                    text = page.get_text("text") or ""
                    text = self._normalize_text(text)

                    if text.strip():
                        pages.append(
                            {
                                "page": index,
                                "text": text,
                            }
                        )
                document.close()
            except Exception:
                pages = []

        return pages

    def _normalize_text(self, text: str):
        text = text.replace("\u2022", "•")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _canonicalize(self, value: str):
        value = value.strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value

    def _match_standard_header(self, line: str):
        normalized = self._canonicalize(line)

        for family, labels in self.config.get("section_headers", {}).items():
            for label in labels:
                if normalized == self._canonicalize(label):
                    return family

        return None

    def _split_standard_sections(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: dict[str, list[str]] = {}
        found_order: list[str] = []

        current_section = None

        for line in lines:
            matched_section = self._match_standard_header(line)

            if matched_section:
                current_section = matched_section
                sections.setdefault(current_section, [])

                if matched_section not in found_order:
                    found_order.append(matched_section)

                continue

            if current_section:
                sections[current_section].append(line)

        clean_sections = {}
        for key, values in sections.items():
            section_text = "\n".join(values).strip()
            if section_text:
                clean_sections[key] = section_text

        return {
            "sections": clean_sections,
            "found_order": found_order,
        }

    def _validate_standard_sections(self, sections: dict[str, str], found_order: list[str]):
        required_sections = self.config.get("required_sections", [])
        missing_required = [
            section for section in required_sections if not sections.get(section)
        ]

        if self.config.get("reject_if_missing_required", True) and missing_required:
            return {
                "valid": False,
                "message": (
                    "PDF does not match the required standard CV format. "
                    f"Missing sections: {', '.join(missing_required)}"
                ),
            }

        expected_order = self.config.get("section_order", [])
        expected_positions = {name: idx for idx, name in enumerate(expected_order)}

        last_position = -1
        for section_name in found_order:
            if section_name not in expected_positions:
                continue

            current_position = expected_positions[section_name]
            if current_position < last_position:
                return {
                    "valid": False,
                    "message": "PDF sections do not follow the expected standard order",
                }

            last_position = current_position

        return {
            "valid": True,
            "message": "Standard CV format validated successfully",
        }

    def _build_section_blocks(self, sections: dict[str, str]):
        blocks = []
        section_order = self.config.get("section_order", [])

        for index, section_name in enumerate(section_order, start=1):
            if section_name not in sections:
                continue

            header = self._get_primary_header_label(section_name)
            block_text = f"{header}\n{sections[section_name]}".strip()

            blocks.append(
                {
                    "page": 1,
                    "column": "single",
                    "top": index * 100,
                    "x0": 0,
                    "section": section_name,
                    "text": block_text,
                }
            )

        return blocks

    def _get_primary_header_label(self, section_name: str):
        labels = self.config.get("section_headers", {}).get(section_name, [])
        if not labels:
            return section_name.upper()
        return labels[0].upper()

    def _build_canonical_text(self, sections: dict[str, str]):
        parts = []
        section_order = self.config.get("section_order", [])

        for section_name in section_order:
            if section_name not in sections:
                continue

            header = self._get_primary_header_label(section_name)
            parts.append(header)
            parts.append(sections[section_name])

        return "\n\n".join(parts).strip()

    def _calculate_quality_score(self, sections: dict[str, str], text: str):
        score = 0

        required_sections = self.config.get("required_sections", [])
        optional_sections = self.config.get("optional_sections", [])

        score += len([s for s in required_sections if s in sections]) * 20
        score += len([s for s in optional_sections if s in sections]) * 5

        if len(text) >= 500:
            score += 10
        if len(text) >= 1000:
            score += 10

        return score