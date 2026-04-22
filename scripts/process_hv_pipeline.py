from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


# =========================
# CONFIGURACIÓN AJUSTABLE
# =========================

DEFAULT_BASE_URL = "http://127.0.0.1:5000"

# Ajusta estos paths si alguno en tu backend tiene otro nombre real.
UPLOAD_BATCH_PATH = "/api/v1/candidates/upload-batch"
EXTRACT_TEXT_PATH = "/api/v1/candidate-sources/{source_id}/extract-text"
NORMALIZE_PROFILE_PATH = "/api/v1/candidate-sources/{source_id}/normalize-profile"
PSEUDONYMIZE_PATH = "/api/v1/candidate-sources/{source_id}/pseudonymize"
LABEL_CANDIDATE_PATH = "/api/v1/job-profiles/{job_profile_id}/candidates/{candidate_id}/label"
BUILD_DATASET_PATH = "/api/v1/job-profiles/{job_profile_id}/datasets/build"

DEFAULT_BATCH_SIZE = 10
REQUEST_TIMEOUT = 180


# =========================
# MODELOS AUXILIARES
# =========================

@dataclass
class CandidatePipelineResult:
    filename: str
    candidate_id: str | None = None
    source_id: str | None = None
    pseudonym_code: str | None = None
    upload_ok: bool = False
    extract_ok: bool = False
    normalize_ok: bool = False
    pseudonymize_ok: bool = False
    error: str | None = None
    normalized_text: str | None = None


# =========================
# CLIENTE API
# =========================

class ApiClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def post_json(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.post(
            url,
            json=body or {},
            timeout=REQUEST_TIMEOUT,
        )
        return self._handle_response(response)

    def post_files(self, path: str, files_payload: list[tuple[str, Any]]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.post(
            url,
            files=files_payload,
            timeout=REQUEST_TIMEOUT,
        )
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError(f"Respuesta no JSON: {response.text}")

        if response.status_code >= 400 or not payload.get("success", False):
            error = payload.get("error") or {}
            message = error.get("message") or payload.get("message") or response.text
            raise RuntimeError(message)

        return payload


# =========================
# UTILIDADES
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted([p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def truncate_text(value: str | None, max_len: int = 500) -> str:
    if not value:
        return ""
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_results_csv(path: Path, rows: list[CandidatePipelineResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "candidate_id",
                "source_id",
                "pseudonym_code",
                "upload_ok",
                "extract_ok",
                "normalize_ok",
                "pseudonymize_ok",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "filename": row.filename,
                    "candidate_id": row.candidate_id,
                    "source_id": row.source_id,
                    "pseudonym_code": row.pseudonym_code,
                    "upload_ok": row.upload_ok,
                    "extract_ok": row.extract_ok,
                    "normalize_ok": row.normalize_ok,
                    "pseudonymize_ok": row.pseudonymize_ok,
                    "error": row.error or "",
                }
            )


def write_label_template_csv(
    path: Path,
    rows: list[CandidatePipelineResult],
    job_profile_id: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "job_profile_id",
                "candidate_id",
                "source_id",
                "pseudonym_code",
                "label",
                "label_reason",
                "split_name",
                "notes",
                "skills_match",
                "experience_match",
                "education_match",
                "language_match",
                "normalized_preview",
            ],
        )
        writer.writeheader()

        for row in rows:
            if not row.candidate_id:
                continue

            writer.writerow(
                {
                    "job_profile_id": job_profile_id,
                    "candidate_id": row.candidate_id,
                    "source_id": row.source_id or "",
                    "pseudonym_code": row.pseudonym_code or "",
                    "label": "",
                    "label_reason": "",
                    "split_name": "train",
                    "notes": "",
                    "skills_match": "",
                    "experience_match": "",
                    "education_match": "",
                    "language_match": "",
                    "normalized_preview": truncate_text(row.normalized_text, 800),
                }
            )


# =========================
# PIPELINE PRINCIPAL
# =========================

def upload_batch(
    client: ApiClient,
    pdf_batch: list[Path],
) -> list[CandidatePipelineResult]:
    file_handles = []
    files_payload = []
    results: list[CandidatePipelineResult] = []

    try:
        for pdf_path in pdf_batch:
            f = pdf_path.open("rb")
            file_handles.append(f)
            files_payload.append(("files", (pdf_path.name, f, "application/pdf")))

        response = client.post_files(UPLOAD_BATCH_PATH, files_payload)

        items = response.get("data") or []
        for item in items:
            filename = ""
            candidate_id = None
            source_id = None
            upload_ok = bool(item.get("success"))

            if item.get("result"):
                result = item["result"]
                filename = (
                    result.get("candidate_source", {}).get("original_filename")
                    or result.get("candidate", {}).get("display_name")
                    or ""
                )
                candidate_id = result.get("candidate", {}).get("id")
                source_id = result.get("candidate_source", {}).get("id")

            results.append(
                CandidatePipelineResult(
                    filename=filename,
                    candidate_id=candidate_id,
                    source_id=source_id,
                    upload_ok=upload_ok,
                    error=None if upload_ok else item.get("error"),
                )
            )

        return results

    finally:
        for f in file_handles:
            f.close()


def process_candidate_source(
    client: ApiClient,
    row: CandidatePipelineResult,
    sleep_seconds: float = 0.0,
) -> CandidatePipelineResult:
    if not row.source_id:
        row.error = row.error or "source_id no disponible después de la carga"
        return row

    try:
        extract_resp = client.post_json(
            EXTRACT_TEXT_PATH.format(source_id=row.source_id)
        )
        row.extract_ok = True
        if sleep_seconds:
            time.sleep(sleep_seconds)

        normalize_resp = client.post_json(
            NORMALIZE_PROFILE_PATH.format(source_id=row.source_id)
        )
        row.normalize_ok = True

        candidate_profile = (normalize_resp.get("data") or {}).get("candidate_profile") or {}
        row.pseudonym_code = candidate_profile.get("pseudonym_code")
        row.normalized_text = candidate_profile.get("normalized_text")

        if sleep_seconds:
            time.sleep(sleep_seconds)

        pseudonymize_resp = client.post_json(
            PSEUDONYMIZE_PATH.format(source_id=row.source_id)
        )
        row.pseudonymize_ok = True

        pseudonymized_profile = (pseudonymize_resp.get("data") or {}).get("candidate_profile") or {}
        if pseudonymized_profile.get("pseudonym_code"):
            row.pseudonym_code = pseudonymized_profile["pseudonym_code"]
        if pseudonymized_profile.get("normalized_text"):
            row.normalized_text = pseudonymized_profile["normalized_text"]

        return row

    except Exception as error:
        row.error = str(error)
        return row


def run_ingest_pipeline(
    client: ApiClient,
    pdf_dir: Path,
    output_dir: Path,
    job_profile_id: str | None,
    batch_size: int,
    sleep_seconds: float,
) -> None:
    ensure_dir(output_dir)

    pdf_files = list_pdf_files(pdf_dir)
    if not pdf_files:
        raise RuntimeError(f"No se encontraron PDFs en: {pdf_dir}")

    print(f"Se encontraron {len(pdf_files)} PDF(s).")

    all_results: list[CandidatePipelineResult] = []
    batches = chunked(pdf_files, batch_size)

    for batch_index, batch in enumerate(batches, start=1):
        print(f"\n[Batch {batch_index}/{len(batches)}] Subiendo {len(batch)} archivo(s)...")
        upload_results = upload_batch(client, batch)

        for result in upload_results:
            if not result.upload_ok:
                print(f"  - ERROR subida: {result.filename} -> {result.error}")
            else:
                print(f"  - OK subida: {result.filename} | source_id={result.source_id}")

        for result in upload_results:
            if not result.upload_ok or not result.source_id:
                all_results.append(result)
                continue

            print(f"    Procesando source_id={result.source_id} ({result.filename})...")
            processed = process_candidate_source(client, result, sleep_seconds=sleep_seconds)

            status = (
                f"upload={processed.upload_ok}, "
                f"extract={processed.extract_ok}, "
                f"normalize={processed.normalize_ok}, "
                f"pseudonymize={processed.pseudonymize_ok}"
            )

            if processed.error:
                print(f"    ERROR {processed.filename}: {processed.error}")
            else:
                print(f"    OK {processed.filename}: {status}")

            all_results.append(processed)

    write_results_csv(output_dir / "pipeline_results.csv", all_results)
    write_json(output_dir / "pipeline_results.json", [row.__dict__ for row in all_results])

    if job_profile_id:
        write_label_template_csv(
            output_dir / "label_template.csv",
            all_results,
            job_profile_id=job_profile_id,
        )
        print("\nSe generó label_template.csv para etiquetado manual.")

    successful = [r for r in all_results if r.pseudonymize_ok]
    failed = [r for r in all_results if r.error]

    print("\nResumen:")
    print(f"  Total archivos: {len(all_results)}")
    print(f"  Exitosos hasta seudonimización: {len(successful)}")
    print(f"  Con error: {len(failed)}")
    print(f"  Reportes guardados en: {output_dir}")


def apply_labels_from_csv(client: ApiClient, labels_csv: Path) -> None:
    if not labels_csv.exists():
        raise RuntimeError(f"No existe el archivo: {labels_csv}")

    with labels_csv.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise RuntimeError("El CSV de etiquetas está vacío")

    ok_count = 0
    error_count = 0

    for row in rows:
        job_profile_id = (row.get("job_profile_id") or "").strip()
        candidate_id = (row.get("candidate_id") or "").strip()
        label = (row.get("label") or "").strip()

        if not job_profile_id or not candidate_id or not label:
            continue

        body = {
            "label": label,
            "label_reason": (row.get("label_reason") or "").strip() or None,
            "split_name": (row.get("split_name") or "train").strip() or "train",
            "notes": (row.get("notes") or "").strip() or None,
            "processing_run_id": None,
            "labeled_by": None,
            "rubric": {
                "skills_match": _to_bool_or_none(row.get("skills_match")),
                "experience_match": _to_bool_or_none(row.get("experience_match")),
                "education_match": _to_bool_or_none(row.get("education_match")),
                "language_match": _to_bool_or_none(row.get("language_match")),
            },
        }

        # Limpiar rubric para no mandar nulls innecesarios
        body["rubric"] = {k: v for k, v in body["rubric"].items() if v is not None}

        try:
            client.post_json(
                LABEL_CANDIDATE_PATH.format(
                    job_profile_id=job_profile_id,
                    candidate_id=candidate_id,
                ),
                body=body,
            )
            ok_count += 1
            print(f"OK etiqueta: candidate_id={candidate_id} -> {label}")
        except Exception as error:
            error_count += 1
            print(f"ERROR etiqueta: candidate_id={candidate_id} -> {error}")

    print("\nResumen etiquetado:")
    print(f"  OK: {ok_count}")
    print(f"  ERROR: {error_count}")


def build_dataset(client: ApiClient, job_profile_id: str, export_format: str) -> None:
    response = client.post_json(
        BUILD_DATASET_PATH.format(job_profile_id=job_profile_id),
        body={"format": export_format},
    )

    print(json.dumps(response, ensure_ascii=False, indent=2))


def _to_bool_or_none(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "si", "sí", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline para carga, procesamiento, etiquetado y construcción de dataset."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"URL base del backend. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Token Bearer opcional si tu backend lo requiere.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Sube PDFs y ejecuta extracción, normalización y seudonimización.")
    ingest_parser.add_argument("--pdf-dir", required=True, help="Carpeta con PDFs.")
    ingest_parser.add_argument("--output-dir", default="output_pipeline", help="Carpeta de salida para reportes.")
    ingest_parser.add_argument("--job-profile-id", default=None, help="Si lo envías, también genera label_template.csv.")
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Tamaño de batch. Default: {DEFAULT_BATCH_SIZE}")
    ingest_parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pausa entre pasos por source.")

    labels_parser = subparsers.add_parser("apply-labels", help="Lee un CSV de etiquetas y llama al endpoint de etiquetado.")
    labels_parser.add_argument("--labels-csv", required=True, help="Ruta al CSV con etiquetas.")

    dataset_parser = subparsers.add_parser("build-dataset", help="Construye el dataset exportado para un job_profile.")
    dataset_parser.add_argument("--job-profile-id", required=True, help="ID del job profile.")
    dataset_parser.add_argument("--format", default="csv", choices=["csv", "json"], help="Formato de exportación.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = ApiClient(base_url=args.base_url, token=args.token)

    try:
        if args.command == "ingest":
            run_ingest_pipeline(
                client=client,
                pdf_dir=Path(args.pdf_dir),
                output_dir=Path(args.output_dir),
                job_profile_id=args.job_profile_id,
                batch_size=args.batch_size,
                sleep_seconds=args.sleep_seconds,
            )
            return 0

        if args.command == "apply-labels":
            apply_labels_from_csv(client, Path(args.labels_csv))
            return 0

        if args.command == "build-dataset":
            build_dataset(client, args.job_profile_id, args.format)
            return 0

        parser.print_help()
        return 1

    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())