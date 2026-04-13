import json
import re
import unicodedata
from typing import Any

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.storage_service import StorageService


class CandidateProfileService:
    HEADER_FAMILIES = {
        "summary": [
            "objetivo laboral",
            "objetivo",
            "perfil profesional",
            "perfil",
            "sobre mi",
            "sobre mí",
            "acerca de mi",
            "acerca de mí",
            "resumen profesional",
            "summary",
            "about me",
        ],
        "experience": [
            "experiencia laboral",
            "experiencia profesional",
            "experiencia",
            "trayectoria",
            "experience",
        ],
        "education": [
            "estudios",
            "formacion",
            "formación",
            "educacion",
            "educación",
            "education",
            "academic background",
        ],
        "skills": [
            "conocimientos y habilidades",
            "habilidades y conocimientos",
            "habilidades",
            "competencias",
            "skills",
            "knowledge",
            "conocimientos",
            "otros conocimientos",
        ],
        "languages": [
            "idiomas",
            "languages",
            "language",
        ],
        "references": [
            "referencias",
            "references",
        ],
        "contact": [
            "datos de contacto",
            "contacto",
            "contact information",
        ],
    }

    DATE_RANGE_RE = re.compile(
        r"\b\d{2}-\d{2}-\d{4}\s*-\s*(?:\d{2}-\d{2}-\d{4}|[A-Za-zÁÉÍÓÚáéíóúÑñ ]+)\b"
    )

    LEVEL_RE = re.compile(
        r"\bseniority\s+(basico|b[aá]sico|intermedio|avanzado)\b",
        re.IGNORECASE,
    )

    TECH_NORMALIZATION = {
        "react.js": "react",
        "reactjs": "react",
        "react native": "react_native",
        "reactnative": "react_native",
        "c#": "csharp",
        "c sharp": "csharp",
        "csharp": "csharp",
        "sql server": "sql_server",
        "sqlserver": "sql_server",
        "power bi": "power_bi",
        "powerbi": "power_bi",
        "postgres": "postgresql",
        "postgre sql": "postgresql",
        "microsoft excel": "excel",
        "microsoft word": "word",
        "wp": "wordpress",
    }

    TECH_DISPLAY = {
        "react": "React",
        "react_native": "React Native",
        "python": "Python",
        "java": "Java",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "csharp": "C#",
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "sql_server": "SQL Server",
        "git": "Git",
        "aws": "AWS",
        "azure": "Azure",
        "power_bi": "Power BI",
        "flask": "Flask",
        "django": "Django",
        "excel": "Excel",
        "word": "Word",
        "wordpress": "WordPress",
        "sql": "SQL",
    }

    SKILL_PATTERNS = [
        (r"\bdevops\b", "devops", "DevOps"),
        (r"\bcloud computing\b", "cloud_computing", "Cloud Computing"),
        (r"\bmicroservicios\b", "microservicios", "Microservicios"),
        (r"\bbases? de datos\b", "bases_de_datos", "Bases de datos"),
        (r"\breportes?\b", "reportes", "Reportes"),
        (r"\btrabajo en equipo\b", "trabajo_en_equipo", "Trabajo en equipo"),
        (r"\bcomunicaci[oó]n\b", "comunicacion", "Comunicación"),
        (r"\bliderazgo\b", "liderazgo", "Liderazgo"),
        (r"\ban[aá]lisis\b", "analisis", "Análisis"),
        (r"\badaptabilidad\b", "adaptabilidad", "Adaptabilidad"),
    ]

    LANGUAGE_PATTERNS = {
        "ingles": ["ingles", "inglés", "english"],
        "espanol": ["espanol", "español", "spanish"],
        "frances": ["frances", "francés", "french"],
        "portugues": ["portugues", "portugués", "portuguese"],
    }

    TECHNOLOGY_BLACKLIST = {
        "soltero",
        "contacto",
        "datos",
        "ecuador",
        "quito",
        "pichincha",
        "referencias",
        "referencia",
        "referente",
        "contact",
        "correo",
        "telefono",
        "celular",
        "dni",
        "cedula",
        "cédula",
        "pasante",
        "desarrollador",
        "junior",
        "seniority",
        "intermedio",
        "basico",
        "básico",
        "oral",
        "escrito",
        "natonar",
        "armas",
        "fue",
        "estudios",
        "objetivo",
        "laboral",
        "habilidades",
        "conocimientos",
        "informatica",
        "informática",
        "universidad",
        "sistemas",
        "presente",
        "curso",
        "en curso",
        "escrito intermedio",
        "oral intermedio",
        "ingles",
        "inglés",
        "devops basico",
        "devops básico",
        "cloud computing",
        "microservicios",
    }

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

    def __init__(self):
        self.storage_service = StorageService()
        self.candidate_source_repository = CandidateSourceRepository()
        self.candidate_profile_repository = CandidateProfileRepository()

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

        if not raw_text.strip():
            raise ValidationError("Normalized artifact does not contain usable text")

        sections = self._split_into_sections(raw_text)

        summary_text = self._summarize_section_lines(
            sections.get("summary", []),
            max_lines=6,
            max_chars=800,
        )

        experience_entries = self._extract_experience_entries(
            sections.get("experience", [])
        )
        education_entries = self._extract_education_entries(
            sections.get("education", [])
        )

        technologies = self._extract_technologies(
            skills_lines=sections.get("skills", []),
            summary_text=summary_text,
            source_type=source["source_type"],
        )
        skills = self._extract_skills(
            skills_lines=sections.get("skills", []),
            summary_text=summary_text,
            source_type=source["source_type"],
        )
        languages = self._extract_languages(
            text=raw_text,
            language_lines=sections.get("languages", []) + sections.get("skills", []),
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
            },
        }

    def get_profile_by_source(self, source_id: str):
        profile = self.candidate_profile_repository.get_by_source_id(source_id)

        if not profile:
            raise NotFoundError("Candidate profile not found")

        return profile

    def _load_artifact(self, source: dict[str, Any]):
        raw_bytes = self.storage_service.download_file(
            source["normalized_bucket"],
            source["normalized_path"],
        )
        return json.loads(raw_bytes.decode("utf-8"))

    def _split_into_sections(self, text: str):
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        sections = {key: [] for key in self.HEADER_FAMILIES.keys()}
        current_section = "summary"

        for line in lines:
            matched = self._match_header_family(line)
            if matched:
                current_section = matched
                continue

            sections.setdefault(current_section, []).append(line)

        return sections

    def _match_header_family(self, line: str):
        normalized_line = self._canonicalize(line)
        normalized_line = re.sub(r"[:\-]+$", "", normalized_line).strip()

        if len(normalized_line.split()) > 5:
            return None

        for family, variants in self.HEADER_FAMILIES.items():
            for variant in variants:
                if normalized_line == self._canonicalize(variant):
                    return family

        return None

    def _extract_experience_entries(self, lines: list[str]):
        if not lines:
            return []

        date_indexes = [
            idx for idx, line in enumerate(lines) if self.DATE_RANGE_RE.search(line)
        ]

        if not date_indexes:
            joined = " ".join(lines[:6]).strip()
            return [{
                "title": None,
                "organization": None,
                "period": None,
                "description": joined[:500],
            }] if joined else []

        entries = []
        for pos, date_idx in enumerate(date_indexes):
            next_date_idx = (
                date_indexes[pos + 1] if pos + 1 < len(date_indexes) else len(lines)
            )

            title = lines[date_idx - 2] if date_idx - 2 >= 0 else None
            organization = lines[date_idx - 1] if date_idx - 1 >= 0 else None
            period = lines[date_idx]

            description_end = (
                next_date_idx - 2 if next_date_idx - date_idx >= 3 else next_date_idx
            )
            description_lines = lines[date_idx + 1:description_end]
            description = " ".join(description_lines).strip()

            entries.append({
                "title": title,
                "organization": organization,
                "period": period,
                "description": description[:600] if description else None,
            })

        return entries[:10]

    def _extract_education_entries(self, lines: list[str]):
        if not lines:
            return []

        entries = []

        if len(lines) >= 3:
            degree = lines[0]
            institution = lines[1] if len(lines) > 1 else None
            level = lines[2] if len(lines) > 2 and not self.DATE_RANGE_RE.search(lines[2]) else None
            period = None

            for line in lines[2:]:
                if self.DATE_RANGE_RE.search(line):
                    period = line
                    break

            entries.append({
                "degree": degree,
                "institution": institution,
                "level": level,
                "period": period,
            })
            return entries

        return [{
            "degree": lines[0] if len(lines) > 0 else None,
            "institution": lines[1] if len(lines) > 1 else None,
            "level": None,
            "period": None,
        }]

    def _extract_technologies(
        self,
        skills_lines: list[str],
        summary_text: str,
        source_type: str,
    ):
        raw_candidates = set()

        for line in skills_lines:
            cleaned_line = self._strip_seniority(line)
            normalized_line = self._canonicalize(cleaned_line)

            if normalized_line.startswith("ingles:") or normalized_line.startswith("inglés:"):
                continue

            if ":" in cleaned_line:
                left, right = cleaned_line.split(":", 1)
                left = left.strip()
                right = right.strip()

                if self._is_reasonable_term(left) and self._looks_like_technology(left):
                    raw_candidates.add(left)

                for term in self._split_candidate_terms(right):
                    if self._looks_like_technology(term):
                        raw_candidates.add(term)
            else:
                for term in self._split_candidate_terms(cleaned_line):
                    if self._looks_like_technology(term):
                        raw_candidates.add(term)

        for term in self._split_candidate_terms(summary_text):
            if self._looks_like_technology(term):
                raw_candidates.add(term)

        items = {}
        for raw in raw_candidates:
            canonical = self._homologate_technology(raw)
            if not canonical or self._is_blacklisted_term(canonical):
                continue

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

        return list(items.values())

    def _extract_skills(
        self,
        skills_lines: list[str],
        summary_text: str,
        source_type: str,
    ):
        text = "\n".join(skills_lines) + "\n" + summary_text
        normalized = self._canonicalize(text)

        items = {}
        for pattern, canonical, display in self.SKILL_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                items[canonical] = {
                    "name": display,
                    "canonical": canonical,
                    "sources": [source_type],
                }

        return list(items.values())

    def _extract_languages(self, text: str, language_lines: list[str], source_type: str):
        combined_text = text + "\n" + "\n".join(language_lines)
        normalized = self._canonicalize(combined_text)

        items = []
        for canonical, variants in self.LANGUAGE_PATTERNS.items():
            found = any(self._canonicalize(variant) in normalized for variant in variants)
            if not found:
                continue

            level = None
            if re.search(r"\bintermedio\b", normalized):
                level = "intermedio"
            elif re.search(r"\bavanzado\b", normalized):
                level = "avanzado"
            elif re.search(r"\bbasico\b|\bb[aá]sico\b", normalized):
                level = "basico"

            items.append({
                "name": self._display_language(canonical),
                "canonical": canonical,
                "level": level,
                "sources": [source_type],
            })

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

    def _summarize_section_lines(self, lines: list[str], max_lines: int = 5, max_chars: int = 500):
        if not lines:
            return ""

        value = " ".join(line.strip() for line in lines[:max_lines] if line.strip())
        return value[:max_chars].strip()

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

    def _split_candidate_terms(self, text: str):
        if not text:
            return []

        text = self._strip_seniority(text)
        text = re.sub(r"\s+y\s+", ", ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+e\s+", ", ", text, flags=re.IGNORECASE)

        pieces = re.split(r"[,\n;|]+", text)
        terms = []

        for piece in pieces:
            cleaned = piece.strip(" .:-")
            if self._is_reasonable_term(cleaned):
                terms.append(cleaned)

        return terms

    def _strip_seniority(self, text: str):
        return self.LEVEL_RE.sub("", text).strip()

    def _is_reasonable_term(self, value: str):
        if not value:
            return False

        if len(value) < 2:
            return False

        if len(value.split()) > 4:
            return False

        if value.isdigit():
            return False

        return True

    def _looks_like_technology(self, term: str):
        canonical = self._canonicalize(term)

        if not canonical or self._is_blacklisted_term(canonical):
            return False

        if canonical in self.TECH_NORMALIZATION:
            return True

        if canonical in self.TECH_DISPLAY:
            return True

        if canonical in {
            "python",
            "java",
            "javascript",
            "typescript",
            "mysql",
            "git",
            "azure",
            "aws",
            "react",
            "wordpress",
            "flask",
            "django",
            "sql",
        }:
            return True

        if any(symbol in term for symbol in ["#", "+", "."]):
            return True

        if "sql" in canonical:
            return True

        if len(term.split()) <= 2 and (
            any(ch.isupper() for ch in term) or term.isupper()
        ):
            return True

        return False

    def _homologate_technology(self, raw: str):
        canonical = self._canonicalize(raw)
        canonical = self.TECH_NORMALIZATION.get(canonical, canonical)

        if len(canonical.split()) > 4:
            return None

        return canonical

    def _display_technology(self, canonical: str, raw: str):
        if canonical in self.TECH_DISPLAY:
            return self.TECH_DISPLAY[canonical]

        return raw.strip()

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

        for token in re.findall(r"[a-z0-9_#.+]{3,}", normalized):
            if token in self.STOPWORDS:
                continue
            if token in self.TECHNOLOGY_BLACKLIST:
                continue
            tokens.add(token)

        return tokens

    def _is_blacklisted_term(self, value: str):
        canonical = self._canonicalize(value)
        return canonical in self.TECHNOLOGY_BLACKLIST

    def _build_pseudonym_code(self, candidate_id: str):
        return f"CAND-{candidate_id[:6].upper()}"

    def _canonicalize(self, value: str):
        value = value.strip().lower()
        value = unicodedata.normalize("NFKD", value)
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = re.sub(r"\s+", " ", value)
        return value