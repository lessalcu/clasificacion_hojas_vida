import json
import re
from pathlib import Path
from typing import Any

from app.errors.exceptions import NotFoundError, ValidationError
from app.repositories.candidate_pii_repository import CandidatePiiRepository
from app.repositories.candidate_profile_repository import CandidateProfileRepository
from app.repositories.candidate_source_repository import CandidateSourceRepository
from app.services.storage_service import StorageService


class PseudonymizationService:
    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    GENERIC_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
    DATE_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")

    IDENTIFICATION_LINE_RE = re.compile(
        r"(?im)^(dni|cedula|cédula|documento de identidad|identificacion|identificación|ci)\s*:\s*([A-Za-z0-9-]+)\s*$"
    )
    ADDRESS_LINE_RE = re.compile(
        r"(?im)^(direccion|dirección|address|domicilio)\s*:\s*(.+)$"
    )
    FULL_NAME_LINE_RE = re.compile(
        r"(?im)^(nombres y apellidos|nombre completo|apellidos y nombres|nombre)\s*:\s*(.+)$"
    )

    MAJOR_SECTION_RE = re.compile(
        r"(?im)^\s*(?:[IVXLC]+\.\s*)?(perfil|experiencia|formacion|formación|educacion|educación|habilidades|idiomas|idioma extranjero|referencias|cursos)\b"
    )
    REFERENCE_SECTION_RE = re.compile(
        r"(?im)^\s*(?:[IVXLC]+\.\s*)?(referencias?|referencia laboral|referencias laborales|referencias personales|referencia personal)\b"
    )
    PHONE_LABEL_RE = re.compile(
        r"(?i)\b(tel[eé]fono|celular|m[oó]vil|whatsapp|tlfno|tlf|fono|contacto)\b"
    )
    ID_LABEL_RE = re.compile(
        r"(?i)\b(dni|cedula|cédula|documento de identidad|identificacion|identificación|ci)\b"
    )
    ADDRESS_LABEL_RE = re.compile(
        r"(?i)\b(direccion|dirección|address|domicilio)\b"
    )
    HONORIFIC_NAME_RE = re.compile(
        r"(?im)^\s*(sr\.?|sra\.?|srta\.?|ing\.?|dr\.?|dra\.?|lic\.?)\s+([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúÑñ]+(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚáéíóúÑñ]+){1,4})\s*$"
    )
    CERTIFICATE_CONTEXT_RE = re.compile(
        r"(?i)(certificado|aprobaci[oó]n|presidente|capacitaci[oó]n|solutions|soluciones)"
    )

    def __init__(self):
        self.candidate_source_repository = CandidateSourceRepository()
        self.candidate_profile_repository = CandidateProfileRepository()
        self.candidate_pii_repository = CandidatePiiRepository()
        self.storage_service = StorageService()
        self.config = self._load_config()

        self.name_blacklist = set(
            x.lower()
            for x in self.config.get("pseudonymization_name_blacklist", [])
        )
        self.institution_words = set(
            x.lower()
            for x in self.config.get("pseudonymization_institution_words", [])
        )
        self.address_hints = [
            x.lower() for x in self.config.get("pseudonymization_address_hints", [])
        ]

    def pseudonymize_source(self, source_id: str):
        source = self.candidate_source_repository.get_by_id(source_id)
        if not source:
            raise NotFoundError("Candidate source not found")

        profile = self.candidate_profile_repository.get_by_source_id(source_id)
        if not profile:
            raise ValidationError(
                "Candidate profile must exist before pseudonymization"
            )

        if not source.get("normalized_bucket") or not source.get("normalized_path"):
            raise ValidationError("Candidate source does not have normalized artifact")

        artifact = self._load_artifact(source)
        original_raw_text = artifact.get("text", "") or ""
        original_normalized_text = profile.get("normalized_text", "") or ""

        pii = self._detect_pii(original_raw_text)

        pii_record = self.candidate_pii_repository.upsert_by_candidate_id(
            candidate_id=source["candidate_id"],
            payload={
                "candidate_id": source["candidate_id"],
                "source_id": source["id"],
                "full_name": pii["full_name"],
                "emails": pii["emails"],
                "phones": pii["phones"],
                "identifications": pii["identifications"],
                "addresses": pii["addresses"],
            },
        )

        pseudonym_code = profile.get("pseudonym_code") or self._build_pseudonym_code(
            source["candidate_id"]
        )

        additional_names = self._extract_additional_person_names(
            text=original_raw_text,
            candidate_full_name=pii.get("full_name"),
        )

        redacted_raw_text = self._redact_text(
            text=original_raw_text,
            pii=pii,
            additional_names=additional_names,
        )

        redacted_normalized_text = self._redact_text(
            text=original_normalized_text,
            pii=pii,
            additional_names=[],
        )

        updated_profile = self.candidate_profile_repository.update_by_source_id(
            source_id,
            {
                "pseudonym_code": pseudonym_code,
                "raw_text": redacted_raw_text,
                "normalized_text": redacted_normalized_text,
            },
        )

        return {
            "status": "pseudonymized",
            "candidate_profile": updated_profile,
            "candidate_pii": {
                "candidate_id": pii_record["candidate_id"],
                "source_id": pii_record["source_id"],
                "full_name_masked": self._mask_name(pii_record.get("full_name")),
                "emails_masked": [
                    self._mask_email(email)
                    for email in (pii_record.get("emails") or [])
                ],
                "phones_masked": [
                    self._mask_phone(phone)
                    for phone in (pii_record.get("phones") or [])
                ],
                "identifications_masked": [
                    self._mask_identification(value)
                    for value in (pii_record.get("identifications") or [])
                ],
                "addresses_detected": len(pii_record.get("addresses") or []),
            },
            "counts": {
                "emails": len(pii["emails"]),
                "phones": len(pii["phones"]),
                "identifications": len(pii["identifications"]),
                "addresses": len(pii["addresses"]),
                "full_name": 1 if pii["full_name"] else 0,
                "additional_names_redacted": len(additional_names),
            },
        }

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

    def _detect_pii(self, text: str):
        personal_text = self._extract_personal_context_text(text)

        full_name = self._extract_full_name(personal_text, fallback_text=text)

        identifications = self._extract_identifications(personal_text)
        emails = self._extract_emails(personal_text)
        phones = self._extract_phones(personal_text)
        addresses = self._extract_addresses(personal_text)

        return {
            "full_name": full_name,
            "emails": emails,
            "phones": phones,
            "identifications": identifications,
            "addresses": addresses,
        }

    def _extract_personal_context_text(self, text: str):
        lines = text.splitlines()
        collected = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                collected.append(line)
                continue

            if self.MAJOR_SECTION_RE.search(stripped):
                break

            collected.append(line)

        return "\n".join(collected).strip()

    def _extract_full_name(self, text: str, fallback_text: str | None = None):
        match = self.FULL_NAME_LINE_RE.search(text)
        if match:
            candidate = self._clean_value(match.group(2))
            if self._is_valid_candidate_name(candidate):
                return candidate

        candidate = self._find_name_in_text(text)
        if candidate:
            return candidate

        if fallback_text:
            candidate = self._find_name_in_text(fallback_text)
            if candidate:
                return candidate

        return None

    def _extract_emails(self, text: str):
        emails = self.EMAIL_RE.findall(text)
        return self._unique(emails)

    def _extract_phones(self, text: str):
        results = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            if self.ID_LABEL_RE.search(stripped):
                continue

            has_phone_context = bool(self.PHONE_LABEL_RE.search(stripped))
            starts_like_contact_bullet = stripped.startswith(("•", "-", "*", "+"))

            if not has_phone_context and not starts_like_contact_bullet:
                continue

            for match in self.GENERIC_PHONE_RE.finditer(stripped):
                value = match.group(0).strip()

                if self._looks_like_date(value):
                    continue

                digits = re.sub(r"\D", "", value)
                if 7 <= len(digits) <= 15:
                    results.append(value)

        return self._unique(results)

    def _extract_identifications(self, text: str):
        values = []
        for match in self.IDENTIFICATION_LINE_RE.finditer(text):
            values.append(self._clean_value(match.group(2)))
        return self._unique(values)

    def _extract_addresses(self, text: str):
        values = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for idx, stripped in enumerate(lines):
            if self.PHONE_LABEL_RE.search(stripped):
                continue
            if self.ID_LABEL_RE.search(stripped):
                continue
            if "@" in stripped:
                continue

            if re.match(r"(?im)^\s*(direccion|dirección|address|domicilio)\s*:", stripped):
                right = stripped.split(":", 1)[1].strip()
                if right:
                    values.append(self._clean_value(right))
                continue

            lowered = stripped.lower()
            if any(hint in lowered for hint in self.address_hints):
                candidate = stripped

                if idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    if (
                        not self.MAJOR_SECTION_RE.search(nxt)
                        and not self.PHONE_LABEL_RE.search(nxt)
                        and not self.ID_LABEL_RE.search(nxt)
                        and "@" not in nxt
                        and len(nxt) <= 60
                    ):
                        candidate = f"{candidate} {nxt}"

                values.append(self._clean_value(candidate))

        return self._unique(values)

    def _find_name_in_text(self, text: str):
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for idx in range(len(lines) - 1):
            first = lines[idx]
            second = lines[idx + 1]

            if self._looks_like_uppercase_name_line(first) and self._looks_like_uppercase_name_line(second):
                candidate = self._clean_value(f"{first} {second}")
                if self._is_valid_candidate_name(candidate):
                    return candidate

        for line in lines[:20]:
            if "@" in line:
                continue
            if self.GENERIC_PHONE_RE.search(line):
                continue
            if ":" in line:
                continue

            candidate = self._clean_value(line)
            if self._is_valid_candidate_name(candidate):
                return candidate

        return None
    
    def _redact_text(
        self,
        text: str,
        pii: dict[str, Any],
        additional_names: list[str] | None = None,
    ):
        if not text:
            return ""

        redacted = text
        additional_names = additional_names or []

        redacted = self._redact_candidate_full_name_multiline(
            redacted,
            pii.get("full_name"),
        )
        redacted = self._redact_candidate_addresses_multiline(
            redacted,
            pii.get("addresses", []),
        )

        replacements = []

        if pii.get("full_name"):
            for variant in self._build_name_variants(pii["full_name"]):
                replacements.append((variant, "[NAME]"))

        for extra_name in additional_names:
            for variant in self._build_name_variants(extra_name):
                replacements.append((variant, "[NAME]"))

        for value in pii.get("addresses", []):
            replacements.append((value, "[ADDRESS]"))

        for value in pii.get("identifications", []):
            replacements.append((value, "[ID]"))

        for value in pii.get("emails", []):
            replacements.append((value, "[EMAIL]"))

        for value in pii.get("phones", []):
            replacements.append((value, "[PHONE]"))

        replacements = self._dedupe_replacements(replacements)
        replacements.sort(key=lambda item: len(item[0]), reverse=True)

        for raw_value, placeholder in replacements:
            if raw_value:
                redacted = re.sub(
                    re.escape(raw_value),
                    placeholder,
                    redacted,
                    flags=re.IGNORECASE,
                )

        redacted = self.EMAIL_RE.sub("[EMAIL]", redacted)

        redacted_lines = []
        for line in redacted.splitlines():
            if self.ID_LABEL_RE.search(line):
                redacted_lines.append(line)
                continue

            def replace_phone(match):
                value = match.group(0)
                if self._looks_like_date(value):
                    return value

                digits = re.sub(r"\D", "", value)
                if 7 <= len(digits) <= 15:
                    return "[PHONE]"
                return value

            redacted_lines.append(self.GENERIC_PHONE_RE.sub(replace_phone, line))

        redacted = "\n".join(redacted_lines)

        redacted = re.sub(
            r"(?im)\b(dni|cedula|cédula|documento de identidad|identificacion|identificación|ci)\b\s*:\s*[A-Za-z0-9-]+",
            lambda m: f"{m.group(1)}: [ID]",
            redacted,
        )

        fixed_lines = []
        for line in redacted.splitlines():
            stripped = line.strip()

            if self.PHONE_LABEL_RE.search(stripped):
                fixed_lines.append(line)
                continue

            if re.match(r"(?im)^\s*(direccion|dirección|address|domicilio)\s*:", stripped):
                label = stripped.split(":", 1)[0]
                indent = re.match(r"^\s*", line).group(0)
                fixed_lines.append(f"{indent}{label}: [ADDRESS]")
                continue

            fixed_lines.append(line)

        redacted = "\n".join(fixed_lines)
        
        if pii.get("full_name"):
            first_name = pii["full_name"].split()[0]
            redacted = re.sub(
                rf"(?i)\b(mi nombre es)\s+{re.escape(first_name)}\b",
                r"\1 [NAME]",
                redacted,
            )

        return redacted

    def _extract_additional_person_names(
        self,
        text: str,
        candidate_full_name: str | None = None,
    ):
        names = set()

        for name in self._extract_reference_names(text, candidate_full_name):
            names.add(name)

        for name in self._extract_signature_names(text, candidate_full_name):
            names.add(name)

        return sorted(names, key=len, reverse=True)

    def _extract_reference_names(
        self,
        text: str,
        candidate_full_name: str | None = None,
    ):
        lines = text.splitlines()
        names = set()
        in_references = False
        buffer_count = 0

        reference_subheaders = {
            "personales",
            "personal",
            "laborales",
            "laboral",
            "profesionales",
            "profesional",
            "académicas",
            "academicas",
            "académica",
            "academica",
        }

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if in_references:
                    buffer_count += 1
                continue

            if self.REFERENCE_SECTION_RE.search(stripped):
                in_references = True
                buffer_count = 0
                continue

            if in_references and stripped.lower() in reference_subheaders:
                continue

            if in_references and self._is_strong_heading(stripped):
                break

            if not in_references:
                continue

            buffer_count += 1
            if buffer_count > 20:
                break

            if "@" in stripped or self.GENERIC_PHONE_RE.search(stripped):
                continue

            honorific_match = self.HONORIFIC_NAME_RE.match(stripped)
            if honorific_match:
                candidate = self._clean_person_name(honorific_match.group(2))
                if self._is_valid_third_party_name(candidate, candidate_full_name):
                    names.add(candidate)
                continue

            cleaned = self._clean_person_name(stripped)
            if self._is_valid_third_party_name(cleaned, candidate_full_name):
                names.add(cleaned)

        return names

    def _extract_signature_names(
        self,
        text: str,
        candidate_full_name: str | None = None,
    ):
        lines = text.splitlines()
        names = set()

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            honorific_match = self.HONORIFIC_NAME_RE.match(stripped)
            if honorific_match:
                candidate = self._clean_person_name(honorific_match.group(2))
                if self._is_valid_third_party_name(candidate, candidate_full_name):
                    names.add(candidate)

            if not self.CERTIFICATE_CONTEXT_RE.search(stripped):
                continue

            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)

            for nearby in lines[start:end]:
                nearby_stripped = nearby.strip()
                if not nearby_stripped:
                    continue
                if "@" in nearby_stripped or self.GENERIC_PHONE_RE.search(nearby_stripped):
                    continue

                cleaned = self._clean_person_name(nearby_stripped)
                if self._is_valid_third_party_name(cleaned, candidate_full_name):
                    names.add(cleaned)

        return names

    def _is_strong_heading(self, value: str):
        normalized = value.strip()

        if self.MAJOR_SECTION_RE.search(normalized):
            return True

        if re.match(r"^[A-ZÁÉÍÓÚÑ\s]{4,}:?$", normalized):
            return True

        if normalized.upper() == normalized and len(normalized.split()) <= 4:
            return True

        return False

    def _clean_person_name(self, value: str):
        value = re.sub(
            r"(?i)\b(sr\.?|sra\.?|srta\.?|ing\.?|dr\.?|dra\.?|lic\.?)\b",
            "",
            value,
        )
        value = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÑñ\s]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def _is_valid_candidate_name(self, value: str):
        if not value:
            return False

        normalized = value.lower().strip()
        if normalized in self.name_blacklist:
            return False

        parts = value.split()
        if len(parts) < 2 or len(parts) > 5:
            return False

        lower_parts = [part.lower() for part in parts]
        if any(part in self.institution_words for part in lower_parts):
            return False

        if not all(part[:1].isupper() for part in parts):
            return False

        return True

    def _is_valid_third_party_name(
        self,
        value: str,
        candidate_full_name: str | None = None,
    ):
        if not value:
            return False

        normalized = value.lower().strip()
        candidate_normalized = (candidate_full_name or "").lower().strip()

        if candidate_normalized and normalized == candidate_normalized:
            return False

        if normalized in self.name_blacklist:
            return False

        parts = value.split()
        if len(parts) < 2 or len(parts) > 4:
            return False

        lower_parts = [part.lower() for part in parts]
        if any(part in self.institution_words for part in lower_parts):
            return False

        if not all(part[:1].isupper() for part in parts):
            return False

        return True

    def _looks_like_uppercase_name_line(self, value: str):
        original = value.strip()
        if not original:
            return False

        if original.lower() in self.name_blacklist:
            return False

        if ":" in original:
            return False
        if "@" in original:
            return False
        if self.GENERIC_PHONE_RE.search(original):
            return False

        cleaned = re.sub(r"[^A-ZÁÉÍÓÚÑ\s]", " ", original).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        if not cleaned:
            return False

        parts = cleaned.split()
        if len(parts) < 1 or len(parts) > 3:
            return False

        lower_parts = [part.lower() for part in parts]
        if any(part in self.institution_words for part in lower_parts):
            return False

        return all(part.isupper() for part in parts)

    def _looks_like_person_name(self, value: str):
        cleaned = re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÑñ\s]", "", value).strip()
        if not cleaned:
            return False

        parts = cleaned.split()
        if len(parts) < 2 or len(parts) > 5:
            return False

        return all(len(part) >= 2 for part in parts)

    def _redact_candidate_full_name_multiline(self, text: str, full_name: str | None):
        if not text or not full_name:
            return text

        parts = [part for part in self._clean_value(full_name).split() if part]
        if len(parts) < 2:
            return text

        pattern = r"(?i)\b" + r"\s+".join(re.escape(part) for part in parts) + r"\b"
        return re.sub(pattern, "[NAME]", text, flags=re.IGNORECASE)

    def _redact_candidate_addresses_multiline(self, text: str, addresses: list[str]):
        if not text or not addresses:
            return text

        redacted = text

        for address in addresses:
            cleaned = self._clean_value(address)
            parts = [part for part in cleaned.split() if part]
            if len(parts) < 2:
                continue

            pattern = r"(?i)\b" + r"\s+".join(re.escape(part) for part in parts) + r"\b"
            redacted = re.sub(pattern, "[ADDRESS]", redacted, flags=re.IGNORECASE)

        return redacted
    
    def _looks_like_date(self, value: str):
        return bool(self.DATE_RE.fullmatch(value.strip()))

    def _dedupe_replacements(self, replacements: list[tuple[str, str]]):
        seen = set()
        result = []

        for raw_value, placeholder in replacements:
            key = (raw_value.strip().lower(), placeholder)
            if key in seen:
                continue
            seen.add(key)
            result.append((raw_value, placeholder))

        return result

    def _build_name_variants(self, full_name: str):
        full_name = self._clean_value(full_name)
        variants = {
            full_name,
            full_name.upper(),
            full_name.title(),
        }

        parts = full_name.split()
        if len(parts) >= 2:
            first_two = " ".join(parts[:2])
            variants.add(first_two)
            variants.add(first_two.upper())
            variants.add(first_two.title())

        return sorted(variants, key=len, reverse=True)

    def _clean_value(self, value: str):
        return re.sub(r"\s+", " ", value).strip()

    def _unique(self, values: list[str]):
        seen = set()
        result = []

        for value in values:
            normalized = value.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(value.strip())

        return result

    def _build_pseudonym_code(self, candidate_id: str):
        return f"CAND-{candidate_id[:6].upper()}"

    def _mask_name(self, value: str | None):
        if not value:
            return None
        parts = value.split()
        return " ".join(
            [part[0] + "***" if len(part) > 1 else part for part in parts]
        )

    def _mask_email(self, value: str):
        if "@" not in value:
            return value

        local, domain = value.split("@", 1)
        if len(local) <= 2:
            return f"{local[0]}***@{domain}"
        return f"{local[:2]}***@{domain}"

    def _mask_phone(self, value: str):
        digits = re.sub(r"\D", "", value)
        if len(digits) <= 4:
            return "***"
        return "*" * (len(digits) - 4) + digits[-4:]

    def _mask_identification(self, value: str):
        if len(value) <= 3:
            return "***"
        return "*" * (len(value) - 3) + value[-3:]