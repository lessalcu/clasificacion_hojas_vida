import re
import unicodedata
from io import BytesIO
from typing import Any

import pdfplumber
import pymupdf

from app.errors.exceptions import ValidationError

ALLOWED_EXTENSIONS = {"pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

REQUIRED_SECTIONS = [
    "DATOS PERSONALES",
    "PERFIL",
    "FORMACIÓN ACADÉMICA",
    "EXPERIENCIA LABORAL",
    "HABILIDADES Y TECNOLOGÍAS",
    "IDIOMAS",
    "CURSOS",
    "REFERENCIAS",
]

REQUIRED_FIELDS_BY_SECTION = {
    "DATOS PERSONALES": [
        "NOMBRE COMPLETO:",
        "CORREO:",
        "TELÉFONO:",
        "DOCUMENTO DE IDENTIDAD:",
        "DIRECCIÓN:",
    ],
    "PERFIL": [],
    "FORMACIÓN ACADÉMICA": [
        "TÍTULO/NIVEL:",
        "INSTITUCIÓN:",
        "PERIODO:",
        "DETALLE:",
    ],
    "EXPERIENCIA LABORAL": [
        "CARGO:",
        "EMPRESA:",
        "PERIODO:",
        "FUNCIONES:",
    ],
    "HABILIDADES Y TECNOLOGÍAS": [
        "LENGUAJES DE PROGRAMACIÓN:",
        "BASES DE DATOS:",
        "FRAMEWORKS Y LIBRERÍAS:",
        "HERRAMIENTAS:",
        "HABILIDADES:",
    ],
    "IDIOMAS": ["IDIOMA:", "NIVEL:"],
    "CURSOS": ["CURSO:", "INSTITUCIÓN:", "DURACIÓN:"],
    "REFERENCIAS": ["NOMBRE:", "CARGO:", "CONTACTO:"],
}

MIN_NATIVE_TEXT_CHARS = 200
MIN_SECTION_CONTENT_CHARS = 8
CERTIFICATE_KEYWORDS = (
    "CERTIFICADO",
    "CERTIFICATE",
    "DIPLOMA",
    "CONSTANCIA DE APROBACIÓN",
    "CONSTANCIA DE PARTICIPACIÓN",
)


def get_extension(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value.upper()


def _read_file_bytes(file_storage) -> bytes:
    file_storage.stream.seek(0)
    data = file_storage.stream.read()
    file_storage.stream.seek(0)
    return data


def _find_heading_indices(lines: list[str]) -> dict[str, int]:
    normalized_lines = [_normalize(line) for line in lines]
    result: dict[str, int] = {}

    for section in REQUIRED_SECTIONS:
        target = _normalize(section)
        try:
            result[section] = normalized_lines.index(target)
        except ValueError:
            continue

    return result


def _extract_section_text(
    lines: list[str], heading_indices: dict[str, int]
) -> dict[str, str]:
    sections: dict[str, str] = {}

    for index, section in enumerate(REQUIRED_SECTIONS):
        start = heading_indices.get(section)
        if start is None:
            continue

        next_positions = [
            heading_indices[next_section]
            for next_section in REQUIRED_SECTIONS[index + 1 :]
            if next_section in heading_indices and heading_indices[next_section] > start
        ]
        end = min(next_positions) if next_positions else len(lines)
        sections[section] = "\n".join(lines[start + 1 : end]).strip()

    return sections


def _detect_multi_column_pages(doc: pymupdf.Document) -> list[int]:
    """Detecta maquetaciones claramente divididas en dos columnas de texto.

    Se usa una heurística conservadora: solo se marca una página cuando existen
    varios pares de bloques de texto, uno a cada lado de la página, que se solapan
    verticalmente. Esto evita confundir una fotografía de perfil con una columna.
    """
    invalid_pages: list[int] = []

    for page_number, page in enumerate(doc, start=1):
        page_width = float(page.rect.width)
        blocks = []

        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, y1, text, *_ = block
            cleaned = (text or "").strip()
            if len(cleaned) < 18:
                continue

            blocks.append(
                {
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "text": cleaned,
                }
            )

        overlaps = 0
        for i, left in enumerate(blocks):
            for right in blocks[i + 1 :]:
                left_is_left = left["x1"] <= page_width * 0.58
                right_is_right = right["x0"] >= page_width * 0.42
                right_is_left = right["x1"] <= page_width * 0.58
                left_is_right = left["x0"] >= page_width * 0.42

                if not (
                    (left_is_left and right_is_right)
                    or (right_is_left and left_is_right)
                ):
                    continue

                vertical_overlap = min(left["y1"], right["y1"]) - max(
                    left["y0"], right["y0"]
                )
                if vertical_overlap <= 4:
                    continue

                overlaps += 1
                if overlaps >= 2:
                    invalid_pages.append(page_number)
                    break
            if page_number in invalid_pages:
                break

    return invalid_pages


def _detect_image_dominant_pages(doc: pymupdf.Document) -> list[int]:
    """Marca páginas dominadas por una imagen grande (p. ej. escaneos/certificados)."""
    invalid_pages: list[int] = []

    for page_number, page in enumerate(doc, start=1):
        page_area = max(float(page.rect.width * page.rect.height), 1.0)
        native_text = re.sub(r"\s+", "", page.get_text("text", sort=True) or "")

        for image_info in page.get_image_info(xrefs=True):
            bbox = image_info.get("bbox")
            if not bbox:
                continue

            rect = pymupdf.Rect(bbox)
            image_ratio = float(rect.width * rect.height) / page_area

            # La fotografía de perfil del formato de referencia ocupa una porción
            # pequeña de la página. Una imagen que domina la página y casi no tiene
            # texto nativo es indicio de PDF escaneado o anexo gráfico.
            if image_ratio >= 0.35 and len(native_text) < 120:
                invalid_pages.append(page_number)
                break

    return invalid_pages


def _detect_complex_table_pages(pdf_bytes: bytes) -> list[int]:
    """Detecta tablas de varias filas/columnas que rompen el formato lineal."""
    invalid_pages: list[int] = []

    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []

                for table in tables:
                    rows = [row for row in table if row]
                    if len(rows) < 2:
                        continue

                    column_count = max((len(row) for row in rows), default=0)
                    non_empty_cells = sum(
                        1
                        for row in rows
                        for cell in row
                        if cell is not None and str(cell).strip()
                    )

                    if column_count >= 2 and non_empty_cells >= 4:
                        invalid_pages.append(page_number)
                        break
    except Exception:
        # Si pdfplumber no puede analizar la geometría, la validación de texto
        # y estructura sigue funcionando con PyMuPDF.
        return []

    return invalid_pages


def validate_standard_cv_pdf(file_storage) -> dict[str, Any]:
    pdf_bytes = _read_file_bytes(file_storage)
    errors: list[str] = []
    warnings: list[str] = []

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValidationError("El archivo no contiene una firma PDF válida")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValidationError("El PDF está dañado o no se puede abrir") from exc

    try:
        if doc.needs_pass:
            errors.append("El PDF está protegido con contraseña")

        if doc.page_count <= 0:
            errors.append("El PDF no contiene páginas")

        page_texts: list[str] = []
        for page in doc:
            page_texts.append(page.get_text("text", sort=True) or "")

        full_text = "\n".join(page_texts)
        useful_text = re.sub(r"\s+", "", full_text)

        if len(useful_text) < MIN_NATIVE_TEXT_CHARS:
            errors.append(
                "No se detectó suficiente texto digital seleccionable. "
                "Los PDF escaneados o basados en imágenes no son admitidos"
            )

        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        heading_indices = _find_heading_indices(lines)
        missing_sections = [
            section for section in REQUIRED_SECTIONS if section not in heading_indices
        ]

        if missing_sections:
            errors.append(
                "Faltan encabezados obligatorios: " + ", ".join(missing_sections)
            )

        # Las secciones deben aparecer en el mismo orden que el formato estándar.
        present_positions = [
            heading_indices[section]
            for section in REQUIRED_SECTIONS
            if section in heading_indices
        ]
        if len(present_positions) == len(
            REQUIRED_SECTIONS
        ) and present_positions != sorted(present_positions):
            errors.append(
                "Las secciones no están en el orden requerido por el formato estándar"
            )

        section_texts = _extract_section_text(lines, heading_indices)
        missing_fields: dict[str, list[str]] = {}
        empty_sections: list[str] = []

        for section in REQUIRED_SECTIONS:
            section_text = section_texts.get(section, "")
            if (
                section in heading_indices
                and len(re.sub(r"\s+", "", section_text)) < MIN_SECTION_CONTENT_CHARS
            ):
                empty_sections.append(section)

            normalized_section = _normalize(section_text)
            absent = [
                field
                for field in REQUIRED_FIELDS_BY_SECTION.get(section, [])
                if _normalize(field) not in normalized_section
            ]
            if absent and section in heading_indices:
                missing_fields[section] = absent

        if empty_sections:
            errors.append("Hay secciones sin contenido: " + ", ".join(empty_sections))

        if missing_fields:
            field_messages = [
                f"{section}: {', '.join(fields)}"
                for section, fields in missing_fields.items()
            ]
            errors.append(
                "Faltan campos del formato estándar en " + " | ".join(field_messages)
            )

        multi_column_pages = _detect_multi_column_pages(doc)
        if multi_column_pages:
            errors.append(
                "El documento debe utilizar una sola columna. "
                "Se detectó una maquetación multicolumna en página(s): "
                + ", ".join(map(str, multi_column_pages))
            )

        complex_table_pages = _detect_complex_table_pages(pdf_bytes)
        if complex_table_pages:
            errors.append(
                "No se permiten tablas complejas. Se detectaron en página(s): "
                + ", ".join(map(str, complex_table_pages))
            )

        image_dominant_pages = _detect_image_dominant_pages(doc)
        if image_dominant_pages:
            errors.append(
                "Se detectaron páginas dominadas por imágenes y sin suficiente texto nativo "
                "en página(s): " + ", ".join(map(str, image_dominant_pages))
            )

        # No se rechaza una mención textual a una certificación dentro de CURSOS.
        # Solo se considera anexo cuando, después de REFERENCIAS, aparece un
        # encabezado independiente típico de certificado/diploma.
        references_index = heading_indices.get("REFERENCIAS")
        if references_index is not None:
            trailing_lines = [
                _normalize(line) for line in lines[references_index + 1 :]
            ]
            certificate_hits = [
                keyword for keyword in CERTIFICATE_KEYWORDS if keyword in trailing_lines
            ]
            if certificate_hits:
                errors.append(
                    "No se permiten certificados o diplomas anexos dentro de la hoja de vida"
                )

        return {
            "valid": len(errors) == 0,
            "page_count": doc.page_count,
            "text_length": len(full_text.strip()),
            "required_sections": REQUIRED_SECTIONS,
            "sections_found": [
                section for section in REQUIRED_SECTIONS if section in heading_indices
            ],
            "missing_sections": missing_sections,
            "missing_fields": missing_fields,
            "multi_column_pages": multi_column_pages,
            "complex_table_pages": complex_table_pages,
            "image_dominant_pages": image_dominant_pages,
            "errors": errors,
            "warnings": warnings,
        }
    finally:
        doc.close()


def validate_pdf_file(
    file_storage, max_size_bytes: int, validate_structure: bool = True
):
    if file_storage is None:
        raise ValidationError("El archivo es obligatorio")

    filename = file_storage.filename or ""
    if not filename.strip():
        raise ValidationError("El nombre del archivo es obligatorio")

    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise ValidationError("Solo se permiten archivos PDF")

    mime_type = file_storage.mimetype or ""
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValidationError("Tipo MIME inválido. Se esperaba application/pdf")

    file_storage.stream.seek(0, 2)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size <= 0:
        raise ValidationError("El archivo está vacío")

    if size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValidationError(
            f"El archivo supera el tamaño máximo permitido de {max_mb:g} MB"
        )

    result: dict[str, Any] = {
        "original_filename": filename,
        "file_extension": extension,
        "mime_type": mime_type,
        "file_size_bytes": size,
    }

    if validate_structure:
        document_validation = validate_standard_cv_pdf(file_storage)
        result["document_validation"] = document_validation

        if not document_validation["valid"]:
            raise ValidationError("; ".join(document_validation["errors"]))

    return result
