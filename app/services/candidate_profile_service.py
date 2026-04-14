import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.storage_service import StorageService


class CandidateProfileService:
    DATE_RANGE_RE = re.compile(
        r"\b\d{2}-\d{2}-\d{4}\s*[-–]\s*(?:\d{2}-\d{2}-\d{4}|[A-Za-zÁÉÍÓÚáéíóúÑñ ]+)\b"
    )
    YEAR_RANGE_RE = re.compile(
        r"\b(?:19|20)\d{2}\s*[-–]\s*(?:presente|actualidad|(?:19|20)\d{2})\b",
        re.IGNORECASE,
    )
    MONTH_YEAR_RE = re.compile(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        re.IGNORECASE,
    )
    MONTH_RANGE_RE = re.compile(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s*[-–]\s*(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        re.IGNORECASE,
    )
    CEFR_LEVEL_RE = re.compile(r"\b([ABC])\s*([12])\b", re.IGNORECASE)
    LEVEL_RE = re.compile(
        r"\bseniority\s+(basico|b[aá]sico|intermedio|avanzado)\b",
        re.IGNORECASE,
    )

    STOPWORDS = {
        "de",
        "la",
        "el",
        "en",
        "y",
        "a",
        "con",
        "para",
        "del",
        "los",
        "las",
        "una",
        "un",
        "que",
        "por",
        "al",
        "mi",
        "mis",
        "su",
        "sus",
        "como",
        "qué",
        "where",
        "with",
        "the",
        "and",
        "or",
    }

    EDUCATION_DEGREE_HINTS = [
        "ingenieria",
        "ingeniería",
        "bachillerato",
        "licenciatura",
        "maestria",
        "maestría",
        "doctorado",
        "tecnologia",
        "tecnología",
        "tecnico",
        "técnico",
        "diplomado",
        "bootcamp",
        "curso",
        "certificacion",
        "certificación",
    ]

    INSTITUTION_HINTS = [
        "universidad",
        "instituto",
        "escuela",
        "colegio",
        "unidad educativa",
        "school",
        "academy",
        "academia",
    ]

    def __init__(self):
        self.storage_service = StorageService()
        self.candidate_source_repository = CandidateSourceRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.config = self._load_config()

    def generate_profile_from_source(self, source_id: str):
        source = self.candidate_source_repository.get_by_id(source_id)

        if not source:
            raise NotFoundError("Candidate source not found")

        if source.get("extraction_status") != "processed":
            raise ValidationError(
                "Candidate source must be processed before normalization"
            )

        if not source.get("normalized_bucket") or not source.get("normalized_path"):
            raise ValidationError(
                "Candidate source does not have a normalized artifact"
            )

        artifact = self._load_artifact(source)
        raw_text = artifact.get("text", "") or ""
        blocks = artifact.get("blocks", []) or []

        if not raw_text.strip() and not blocks:
            raise ValidationError("Normalized artifact does not contain usable content")

        sections = (
            self._extract_sections_from_blocks(blocks)
            if blocks
            else self._extract_sections_from_text(raw_text)
        )

        summary_text = self._summarize_blocks(sections.get("summary", []))
        experience_entries = self._extract_experience_entries_from_blocks(
            sections.get("experience", [])
        )
        education_entries = self._extract_education_entries_from_blocks(
            sections.get("education", [])
        )

        technologies = self._extract_technologies_from_blocks(
            skills_blocks=sections.get("skills", []),
            summary_blocks=sections.get("summary", []),
            experience_blocks=sections.get("experience", []),
            source_type=source["source_type"],
        )
        skills = self._extract_skills_from_blocks(
            skills_blocks=sections.get("skills", []),
            summary_blocks=sections.get("summary", []),
            experience_blocks=sections.get("experience", []),
            source_type=source["source_type"],
        )
        languages = self._extract_languages_from_blocks(
            raw_text=raw_text,
            language_blocks=sections.get("languages", []),
            skills_blocks=sections.get("skills", []),
            source_type=source["source_type"],
        )

        experience_summary = self._build_experience_summary(experience_entries)
        education_summary = self._build_education_summary(education_entries)

        keywords = self._build_keywords(
            technologies=technologies,
            skills=skills,
            languages=languages,
            experience_entries=experience_entries,
            education_entries=education_entries,
        )

        normalized_text = self._build_normalized_text(
            technologies=technologies,
            skills=skills,
            languages=languages,
            experience_entries=experience_entries,
            education_entries=education_entries,
            summary_text=summary_text,
        )

        payload = {
            "candidate_id": source["candidate_id"],
            "source_id": source["id"],
            "source_type": source["source_type"],
            "pseudonym_code": self._build_pseudonym_code(source["candidate_id"]),
            "skills": skills,
            "technologies": technologies,
            "experience_summary": experience_summary,
            "education_summary": education_summary,
            "languages": languages,
            "keywords": keywords,
            "raw_text": raw_text,
            "normalized_text": normalized_text,
        }

        existing = self.candidate_profile_repository.get_by_source_id(source["id"])

        if existing:
            profile = self.candidate_profile_repository.update_by_source_id(
                source["id"],
                payload,
            )
        else:
            profile = self.candidate_profile_repository.create(payload)

        return {
            "status": "generated",
            "candidate_profile": profile,
            "sections_found": {
                "summary": bool(sections.get("summary")),
                "experience": bool(sections.get("experience")),
                "education": bool(sections.get("education")),
                "skills": bool(sections.get("skills")),
                "languages": bool(sections.get("languages")),
                "courses": bool(sections.get("courses")),
            },
        }

    def get_profile_by_source(self, source_id: str):
        profile = self.candidate_profile_repository.get_by_source_id(source_id)

        if not profile:
            raise NotFoundError("Candidate profile not found")

        return profile

    def _load_config(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "normalization_config.json"
        )

        if not config_path.exists():
            raise ValidationError(
                f"Normalization config not found at: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_artifact(self, source: dict[str, Any]):
        raw_bytes = self.storage_service.download_file(
            source["normalized_bucket"],
            source["normalized_path"],
        )
        return json.loads(raw_bytes.decode("utf-8"))

    def _extract_sections_from_blocks(self, blocks: list[dict[str, Any]]):
        sections = {key: [] for key in self.config["section_hints"].keys()}

        ordered = sorted(
            blocks,
            key=lambda block: (
                int(block.get("page", 1)),
                0 if block.get("column", "left") == "left" else 1,
                float(block.get("top", 0)),
                float(block.get("x0", 0)),
            ),
        )

        current_by_scope: dict[tuple[int, str], str] = {}

        for block in ordered:
            page = int(block.get("page", 1))
            column = str(block.get("column", "left"))
            scope = (page, column)

            text = block.get("text", "").strip()
            if not text:
                continue

            family, body_text = self._split_header_and_body_from_block(text)

            if family:
                current_by_scope[scope] = family

                if body_text:
                    body_block = dict(block)
                    body_block["text"] = body_text
                    sections[family].append(body_block)

                continue

            if self._looks_like_skills_block(text):
                sections["skills"].append(block)
                current_by_scope[scope] = "skills"
                continue

            current_family = current_by_scope.get(scope)
            if current_family:
                sections[current_family].append(block)

        return sections

    def _extract_sections_from_text(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = {key: [] for key in self.config["section_hints"].keys()}
        current_section = "summary"

        for line in lines:
            family = self._match_header_family_from_block(line)
            if family:
                current_section = family
                continue

            sections[current_section].append(
                {
                    "page": 1,
                    "column": "left",
                    "top": 0,
                    "x0": 0,
                    "text": line,
                }
            )

        return sections

    def _split_header_and_body_from_block(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None, ""

        if len(lines) >= 2:
            two_line_candidate = f"{lines[0]} {lines[1]}"
            family = self._match_header_family_from_block(two_line_candidate)
            if family:
                body = "\n".join(lines[2:]).strip()
                return family, body

        family = self._match_header_family_from_block(lines[0])
        if family:
            body = "\n".join(lines[1:]).strip()
            return family, body

        return None, text.strip()
    
    def _looks_like_skills_block(self, text: str):
        if not text:
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        label_hits = 0
        known_labels = {self._canonicalize(x) for x in self.config["list_labels"]}

        for line in lines:
            normalized_line = self._canonicalize(line.strip(" :"))

            if normalized_line in known_labels:
                label_hits += 1
                continue

            if ":" in line:
                left = line.split(":", 1)[0].strip()
                if self._canonicalize(left) in known_labels:
                    label_hits += 1

        tech_hits = len(self._scan_known_technologies(text))

        if label_hits >= 1 and tech_hits >= 2:
            return True

        if tech_hits >= 4:
            return True

        return False
    
    def _match_header_family_from_block(self, text: str):
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        candidates = []
        candidates.append(lines[0])

        if len(lines) >= 2:
            candidates.append(f"{lines[0]} {lines[1]}")

        for candidate in candidates:
            normalized = self._canonicalize(candidate)
            normalized = re.sub(r"[:\-]+$", "", normalized).strip()

            if len(normalized.split()) > 6:
                continue

            for family, variants in self.config["section_hints"].items():
                for variant in variants:
                    if normalized == self._canonicalize(variant):
                        return family

        return None

    def _summarize_blocks(
        self,
        blocks: list[dict[str, Any]],
        max_blocks: int = 3,
        max_chars: int = 800,
    ):
        if not blocks:
            return ""

        text = " ".join(
            block.get("text", "").replace("\n", " ").strip()
            for block in blocks[:max_blocks]
            if block.get("text", "").strip()
        )
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars]

    def _extract_experience_entries_from_blocks(self, blocks: list[dict[str, Any]]):
        if not blocks:
            return []

        all_lines = []
        for block in blocks:
            lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
            all_lines.extend(lines)

        if not all_lines:
            return []

        entries: list[dict[str, Any]] = []
        current = None
        index = 0

        while index < len(all_lines):
            line = all_lines[index]

            if self._looks_like_experience_title(line):
                if current and self._entry_has_content(current):
                    entries.append(self._normalize_entry(current))

                current = {
                    "title": self._single_line(line),
                    "organization": None,
                    "period": None,
                    "description": None,
                }

                index += 1

                if index < len(all_lines) and self._looks_like_period(all_lines[index]):
                    current["period"] = self._single_line(all_lines[index])
                    index += 1

                while index < len(all_lines):
                    next_line = all_lines[index]

                    if self._looks_like_experience_title(next_line):
                        break

                    if current.get("period") is None and self._looks_like_period(next_line):
                        current["period"] = self._single_line(next_line)
                    else:
                        current["description"] = self._append_text(
                            current.get("description"),
                            next_line,
                        )

                    index += 1

                continue

            if current is None:
                current = {
                    "title": None,
                    "organization": None,
                    "period": None,
                    "description": None,
                }

            if current.get("period") is None and self._looks_like_period(line):
                current["period"] = self._single_line(line)
            else:
                current["description"] = self._append_text(
                    current.get("description"),
                    line,
                )

            index += 1

        if current and self._entry_has_content(current):
            entries.append(self._normalize_entry(current))

        return self._merge_fragmented_experience_entries(entries[:10])

    def _extract_education_entries_from_blocks(self, blocks: list[dict[str, Any]]):
        if not blocks:
            return []

        all_lines = []
        for block in blocks:
            lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
            all_lines.extend(lines)

        if not all_lines:
            return []

        entries = []
        current = None

        for line in all_lines:
            if self._looks_like_education_degree(line):
                if current and self._education_entry_has_content(current):
                    entries.append(self._normalize_education_entry(current))

                current = {
                    "degree": self._single_line(line),
                    "institution": None,
                    "level": None,
                    "period": None,
                }
                continue

            if current is None:
                current = {
                    "degree": self._single_line(line),
                    "institution": None,
                    "level": None,
                    "period": None,
                }
                continue

            if self._looks_like_institution(line) and not current.get("institution"):
                current["institution"] = self._single_line(line)
            elif self._looks_like_period(line) and not current.get("period"):
                current["period"] = self._single_line(line)
            else:
                current["level"] = self._append_text(current.get("level"), line)

        if current and self._education_entry_has_content(current):
            entries.append(self._normalize_education_entry(current))

        return entries[:10]

    def _extract_technologies_from_blocks(
        self,
        skills_blocks: list[dict[str, Any]],
        summary_blocks: list[dict[str, Any]],
        experience_blocks: list[dict[str, Any]],
        source_type: str,
    ):
        items: dict[str, dict[str, Any]] = {}
        normalized_labels = {
            self._canonicalize(label) for label in self.config["list_labels"]
        }

        for block in skills_blocks:
            block_text = block.get("text", "")
            for line in block_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                cleaned_line = self._strip_seniority(line)

                if ":" in cleaned_line:
                    left, right = cleaned_line.split(":", 1)
                    left = left.strip()
                    right = right.strip()
                    left_norm = self._canonicalize(left)

                    if left_norm in normalized_labels:
                        for term in self._extract_list_terms(right):
                            self._add_technology_item(items, term, source_type)
                    else:
                        self._add_technology_item(items, left, source_type)
                        for term in self._extract_list_terms(right):
                            self._add_technology_item(items, term, source_type)
                else:
                    for term in self._extract_list_terms(cleaned_line):
                        self._add_technology_item(items, term, source_type)

        scan_text = "\n".join(
            block.get("text", "")
            for block in (summary_blocks + experience_blocks + skills_blocks)
        )
        for raw in self._scan_known_technologies(scan_text):
            self._add_technology_item(items, raw, source_type)

        return list(items.values())

    def _parse_language_line(self, line: str, source_type: str):
        if not line:
            return None

        normalized_line = self._canonicalize(line)

        for canonical, variants in self.config["language_aliases"].items():
            if any(self._canonicalize(variant) in normalized_line for variant in variants):
                return {
                    "name": self._display_language(canonical),
                    "canonical": canonical,
                    "level": self._infer_language_level(line),
                    "sources": [source_type],
                }

        return None
    
    def _infer_language_level(self, text: str):
        normalized = self._canonicalize(text)

        if "nativo" in normalized or "native" in normalized:
            return "nativo"

        cefr_match = self.CEFR_LEVEL_RE.search(text)
        if cefr_match:
            level = f"{cefr_match.group(1).upper()}{cefr_match.group(2)}"
            if level in {"A1", "A2"}:
                return "basico"
            if level in {"B1", "B2"}:
                return "intermedio"
            if level in {"C1", "C2"}:
                return "avanzado"

        for level_key, aliases in self.config["language_levels"].items():
            if any(self._canonicalize(alias) in normalized for alias in aliases):
                return level_key

        return None

    def _extract_skills_from_blocks(
        self,
        skills_blocks: list[dict[str, Any]],
        summary_blocks: list[dict[str, Any]],
        experience_blocks: list[dict[str, Any]],
        source_type: str,
    ):
        text = "\n".join(
            block.get("text", "")
            for block in (skills_blocks + summary_blocks + experience_blocks)
        )
        normalized = self._canonicalize(text)

        items = {}
        for item in self.config["skill_terms"]:
            terms = item["terms"]
            found = any(self._canonicalize(term) in normalized for term in terms)
            if found:
                items[item["canonical"]] = {
                    "name": item["display"],
                    "canonical": item["canonical"],
                    "sources": [source_type],
                }

        return list(items.values())

    def _extract_languages_from_blocks(
    self,
    raw_text: str,
    language_blocks: list[dict[str, Any]],
    skills_blocks: list[dict[str, Any]],
    source_type: str,
    ):
        parsed_items = []

        for block in language_blocks:
            lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
            for line in lines:
                parsed = self._parse_language_line(line, source_type)
                if parsed:
                    parsed_items.append(parsed)

        if parsed_items:
            deduped = {}
            for item in parsed_items:
                canonical = item["canonical"]
                if canonical not in deduped:
                    deduped[canonical] = item
                else:
                    if item.get("level") and not deduped[canonical].get("level"):
                        deduped[canonical]["level"] = item["level"]
            return list(deduped.values())

        combined_text = raw_text + "\n" + "\n".join(
            block.get("text", "") for block in (language_blocks + skills_blocks)
        )
        normalized = self._canonicalize(combined_text)

        items = []
        for canonical, variants in self.config["language_aliases"].items():
            found = any(
                self._canonicalize(variant) in normalized for variant in variants
            )
            if not found:
                continue

            items.append(
                {
                    "name": self._display_language(canonical),
                    "canonical": canonical,
                    "level": self._infer_language_level(combined_text),
                    "sources": [source_type],
                }
            )

        return items

    def _build_experience_summary(self, entries: list[dict[str, Any]]):
        if not entries:
            return ""

        parts = []
        for entry in entries[:5]:
            fragments = [
                entry.get("title"),
                entry.get("organization"),
                entry.get("period"),
                entry.get("description"),
            ]
            fragments = [frag.strip() for frag in fragments if frag and frag.strip()]
            if fragments:
                parts.append(" | ".join(fragments))

        return " ; ".join(parts)

    def _build_education_summary(self, entries: list[dict[str, Any]]):
        if not entries:
            return ""

        parts = []
        for entry in entries[:5]:
            fragments = [
                entry.get("degree"),
                entry.get("institution"),
                entry.get("level"),
                entry.get("period"),
            ]
            fragments = [frag.strip() for frag in fragments if frag and frag.strip()]
            if fragments:
                parts.append(" | ".join(fragments))

        return " ; ".join(parts)

    def _build_keywords(
        self,
        technologies: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        languages: list[dict[str, Any]],
        experience_entries: list[dict[str, Any]],
        education_entries: list[dict[str, Any]],
    ):
        keywords = set()

        for item in technologies:
            keywords.add(item["canonical"])

        for item in skills:
            keywords.add(item["canonical"])

        for item in languages:
            keywords.add(item["canonical"])

        for entry in experience_entries:
            for value in [entry.get("title"), entry.get("organization")]:
                keywords.update(self._extract_clean_tokens(value))

        for entry in education_entries:
            for value in [entry.get("degree"), entry.get("institution")]:
                keywords.update(self._extract_clean_tokens(value))

        return sorted(list(keywords))[:100]

    def _build_normalized_text(
        self,
        technologies: list[dict[str, Any]],
        skills: list[dict[str, Any]],
        languages: list[dict[str, Any]],
        experience_entries: list[dict[str, Any]],
        education_entries: list[dict[str, Any]],
        summary_text: str,
    ):
        parts = []

        if technologies:
            parts.append(
                "TECNOLOGÍAS: " + ", ".join(item["name"] for item in technologies)
            )

        if skills:
            parts.append(
                "HABILIDADES: " + ", ".join(item["name"] for item in skills)
            )

        if languages:
            language_parts = []
            for item in languages:
                if item.get("level"):
                    language_parts.append(f"{item['name']} ({item['level']})")
                else:
                    language_parts.append(item["name"])
            parts.append("IDIOMAS: " + ", ".join(language_parts))

        if experience_entries:
            parts.append("EXPERIENCIA:")
            for entry in experience_entries[:5]:
                fragments = [
                    entry.get("title"),
                    entry.get("organization"),
                    entry.get("period"),
                ]
                fragments = [frag.strip() for frag in fragments if frag and frag.strip()]
                if entry.get("description"):
                    fragments.append(entry["description"].strip())
                parts.append("- " + " | ".join(fragments))

        if education_entries:
            parts.append("FORMACIÓN:")
            for entry in education_entries[:5]:
                fragments = [
                    entry.get("degree"),
                    entry.get("institution"),
                    entry.get("level"),
                    entry.get("period"),
                ]
                fragments = [frag.strip() for frag in fragments if frag and frag.strip()]
                parts.append("- " + " | ".join(fragments))

        if summary_text:
            parts.append("RESUMEN: " + summary_text)

        return "\n".join(parts).strip()

    def _extract_list_terms(self, text: str):
        if not text:
            return []

        text = self._strip_seniority(text)
        text = re.sub(r"\s+y\s+", ", ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+e\s+", ", ", text, flags=re.IGNORECASE)
        text = re.sub(r"[()]", "", text)

        pieces = re.split(r"[,;/\n]+", text)
        terms = []

        for piece in pieces:
            cleaned = piece.strip(" .:-")
            if not cleaned:
                continue

            subpieces = re.split(r"\.\s+", cleaned)
            for sub in subpieces:
                sub = sub.strip(" .:-")
                if self._is_reasonable_term(sub):
                    terms.append(sub)

        return terms

    def _scan_known_technologies(self, text: str):
        normalized = self._canonicalize(text)
        found = []

        aliases = sorted(
            self.config["technology_aliases"].keys(),
            key=len,
            reverse=True,
        )
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(self._canonicalize(alias)) + r"(?!\w)"
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                found.append(alias)

        return found

    def _add_technology_item(
        self,
        items: dict[str, dict[str, Any]],
        raw: str,
        source_type: str,
    ):
        if not raw:
            return

        canonical = self._homologate_technology(raw)
        if not canonical:
            return

        if self._is_blacklisted_term(canonical):
            return

        if canonical not in items:
            items[canonical] = {
                "name": self._display_technology(canonical, raw),
                "canonical": canonical,
                "sources": [source_type],
                "raw_values": [raw],
            }
        else:
            if raw not in items[canonical]["raw_values"]:
                items[canonical]["raw_values"].append(raw)

    def _looks_like_experience_title(self, text: str, full_block: str | None = None):
        text = self._single_line(text)
        if not text:
            return False

        words = text.split()
        normalized = self._canonicalize(text)

        if len(words) > 6:
            return False

        if self._looks_like_period(text):
            return False

        if ":" in text:
            return False
        if text.endswith(".") or text.endswith(",") or text.endswith(";"):
            return False

        all_section_terms = {
            self._canonicalize(term)
            for values in self.config["section_hints"].values()
            for term in values
        }
        if normalized in all_section_terms:
            return False

        return True

    def _looks_like_education_degree(self, text: str):
        normalized = self._canonicalize(text)
        return any(hint in normalized for hint in self.EDUCATION_DEGREE_HINTS)

    def _looks_like_institution(self, text: str):
        normalized = self._canonicalize(text)
        return any(hint in normalized for hint in self.INSTITUTION_HINTS)

    def _looks_like_period(self, text: str):
        normalized = self._canonicalize(text)

        return bool(
            self.DATE_RANGE_RE.search(text)
            or self.YEAR_RANGE_RE.search(normalized)
            or self.MONTH_RANGE_RE.search(normalized)
            or self.MONTH_YEAR_RE.search(normalized)
        )

    def _find_date_line_index(self, lines: list[str]):
        for idx, line in enumerate(lines):
            if self._looks_like_period(line):
                return idx
        return None

    def _entry_has_content(self, entry: dict[str, Any]):
        return any(
            entry.get(field)
            for field in ["title", "organization", "period", "description"]
        )

    def _education_entry_has_content(self, entry: dict[str, Any]):
        return any(
            entry.get(field)
            for field in ["degree", "institution", "level", "period"]
        )

    def _normalize_entry(self, entry: dict[str, Any]):
        return {
            "title": self._clean_joined_text(entry.get("title")),
            "organization": self._clean_joined_text(entry.get("organization")),
            "period": self._clean_joined_text(entry.get("period")),
            "description": self._clean_joined_text(entry.get("description")),
        }

    def _normalize_education_entry(self, entry: dict[str, Any]):
        return {
            "degree": self._clean_joined_text(entry.get("degree")),
            "institution": self._clean_joined_text(entry.get("institution")),
            "level": self._clean_joined_text(entry.get("level")),
            "period": self._clean_joined_text(entry.get("period")),
        }

    def _merge_fragmented_experience_entries(self, entries: list[dict[str, Any]]):
        if not entries:
            return []

        merged = []

        for entry in entries:
            title = entry.get("title")
            description = entry.get("description")
            period = entry.get("period")
            organization = entry.get("organization")

            if not title and description and merged:
                merged[-1]["description"] = self._append_text(
                    merged[-1].get("description"),
                    description,
                )
                continue

            merged.append(entry)

        return merged

    def _append_text(self, original: str | None, new_value: str | None):
        if not new_value:
            return original
        if not original:
            return self._single_line(new_value)
        return f"{original} | {self._single_line(new_value)}"

    def _single_line(self, text: str):
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

    def _clean_joined_text(self, text: str | None):
        if not text:
            return None
        return re.sub(r"\s+", " ", text).strip()

    def _strip_seniority(self, text: str):
        return self.LEVEL_RE.sub("", text).strip()

    def _is_reasonable_term(self, value: str):
        if not value:
            return False

        value = self._single_line(value)

        if len(value) < 2:
            return False
        if len(value.split()) > 4:
            return False
        if value.isdigit():
            return False

        return True

    def _homologate_technology(self, raw: str):
        normalized = self._canonicalize(raw)
        aliases = self.config["technology_aliases"]

        if normalized in aliases:
            return aliases[normalized][0]

        if len(normalized.split()) > 4:
            return None

        blocked = {self._canonicalize(x) for x in self.config["blocked_terms"]}
        if normalized in blocked:
            return None

        return normalized if self._looks_like_generic_technology(raw) else None

    def _display_technology(self, canonical: str, raw: str):
        for _alias, value in self.config["technology_aliases"].items():
            canon, display = value
            if canon == canonical:
                return display
        return self._single_line(raw)

    def _looks_like_generic_technology(self, term: str):
        normalized = self._canonicalize(term)

        if not normalized or self._is_blacklisted_term(normalized):
            return False

        if any(symbol in term for symbol in ["#", "+", "."]):
            return True

        if "sql" in normalized:
            return True

        if len(term.split()) <= 2 and (
            any(ch.isupper() for ch in term) or term.isupper()
        ):
            return True

        return False

    def _display_language(self, canonical: str):
        mapping = {
            "ingles": "Inglés",
            "espanol": "Español",
            "frances": "Francés",
            "portugues": "Portugués",
        }
        return mapping.get(canonical, canonical.title())

    def _extract_clean_tokens(self, value: str | None):
        if not value:
            return set()

        normalized = self._canonicalize(value)
        tokens = set()
        blocked = {self._canonicalize(x) for x in self.config["blocked_terms"]}

        for token in re.findall(r"[a-z0-9_#.+/]{2,}", normalized):
            if token in self.STOPWORDS:
                continue
            if token in blocked:
                continue
            tokens.add(token)

        return tokens

    def _extract_experience_entries_from_blocks(self, blocks: list[dict[str, Any]]):
        if not blocks:
            return []

        all_lines = []
        for block in blocks:
            lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
            all_lines.extend(lines)

        if not all_lines:
            return []

        entries: list[dict[str, Any]] = []
        index = 0

        while index < len(all_lines):
            line = all_lines[index]

            if not self._looks_like_experience_title(line):
                index += 1
                continue

            title = self._single_line(line)
            index += 1

            while index < len(all_lines) and self._looks_like_experience_title_continuation(
                title, all_lines[index]
            ):
                title = f"{title} {self._single_line(all_lines[index])}"
                index += 1

            period = None
            if index < len(all_lines) and self._looks_like_period(all_lines[index]):
                period = self._single_line(all_lines[index])
                index += 1

            description_parts = []

            while index < len(all_lines):
                next_line = all_lines[index]

                if self._looks_like_experience_title(next_line):
                    break

                if period is None and self._looks_like_period(next_line):
                    period = self._single_line(next_line)
                    index += 1
                    continue

                description_parts.append(self._single_line(next_line))
                index += 1

            entries.append(
                self._normalize_entry(
                    {
                        "title": title,
                        "organization": None,
                        "period": period,
                        "description": " ".join(description_parts).strip() or None,
                    }
                )
            )

        return entries[:10]


    def _looks_like_experience_title(self, text: str):
        text = self._single_line(text)
        if not text:
            return False

        words = text.split()
        normalized = self._canonicalize(text)

        if len(words) > 8:
            return False

        if self._looks_like_period(text):
            return False

        if text.endswith(".") or text.endswith(",") or text.endswith(";"):
            return False

        if ":" in text:
            return False

        all_section_terms = {
            self._canonicalize(term)
            for values in self.config["section_hints"].values()
            for term in values
        }
        if normalized in all_section_terms:
            return False

        return True


    def _looks_like_experience_title_continuation(self, current_title: str, text: str):
        text = self._single_line(text)
        if not text:
            return False

        if self._looks_like_period(text):
            return False

        if text.endswith(".") or text.endswith(",") or text.endswith(";"):
            return False

        if ":" in text:
            return False

        if len(text.split()) > 6:
            return False

        current_normalized = self._canonicalize(current_title)

        if current_normalized.endswith((" de", " del", " la", " el")):
            return True

        if "(" in text or ")" in text:
            return True

        if text[0].isupper():
            return True

        return False


    def _extract_education_entries_from_blocks(self, blocks: list[dict[str, Any]]):
        if not blocks:
            return []

        all_lines = []
        for block in blocks:
            lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
            all_lines.extend(lines)

        if not all_lines:
            return []

        entries = []
        current = None

        for line in all_lines:
            if self._looks_like_education_degree(line):
                if current and self._education_entry_has_content(current):
                    entries.append(self._normalize_education_entry(current))

                current = {
                    "degree": self._single_line(line),
                    "institution": None,
                    "level": None,
                    "period": None,
                }
                continue

            if current is None:
                current = {
                    "degree": self._single_line(line),
                    "institution": None,
                    "level": None,
                    "period": None,
                }
                continue

            if self._looks_like_institution(line) and not current.get("institution"):
                current["institution"] = self._single_line(line)
            elif self._looks_like_period(line) and not current.get("period"):
                current["period"] = self._single_line(line)
            else:
                current["level"] = self._append_text(current.get("level"), line)

        if current and self._education_entry_has_content(current):
            entries.append(self._normalize_education_entry(current))

        return entries[:10]


    def _looks_like_education_degree(self, text: str):
        normalized = self._canonicalize(text)

        if any(hint in normalized for hint in self.EDUCATION_DEGREE_HINTS):
            return True

        if normalized.startswith("educacion ") and len(normalized.split()) <= 4:
            return True

        if normalized.startswith("education ") and len(normalized.split()) <= 4:
            return True

        return False

    def _is_blacklisted_term(self, value: str):
        canonical = self._canonicalize(value)
        blocked = {self._canonicalize(x) for x in self.config["blocked_terms"]}
        return canonical in blocked

    def _build_pseudonym_code(self, candidate_id: str):
        return f"CAND-{candidate_id[:6].upper()}"

    def _canonicalize(self, value: str):
        value = value.strip().lower()
        value = unicodedata.normalize("NFKD", value)
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = re.sub(r"\s+", " ", value)
        return value