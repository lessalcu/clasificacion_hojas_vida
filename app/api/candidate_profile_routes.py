from io import BytesIO

from flask import (
    Blueprint,
    jsonify,
    request,
    send_file,
)

from app.services.candidate_profile_service import (
    CandidateProfileService,
)
from app.services.candidate_source_file_service import (
    CandidateSourceFileService,
)

candidate_profile_bp = Blueprint(
    "candidate_profile",
    __name__,
    url_prefix="/api/v1/candidate-sources",
)

service = CandidateProfileService()
file_service = CandidateSourceFileService()


@candidate_profile_bp.post(
    "/<source_id>/normalize-profile"
)
def normalize_profile(source_id: str):
    result = (
        service.generate_profile_from_source(
            source_id
        )
    )

    return jsonify(
        {
            "success": True,
            "message": (
                "Candidate profile generated successfully"
            ),
            "data": result,
        }
    ), 200


@candidate_profile_bp.get(
    "/<source_id>/profile"
)
def get_profile(source_id: str):
    result = service.get_profile_by_source(
        source_id
    )

    return jsonify(
        {
            "success": True,
            "data": result,
        }
    ), 200


@candidate_profile_bp.get(
    "/<source_id>/file"
)
def get_candidate_source_file(
    source_id: str,
):
    result = file_service.get_pdf_file(
        source_id
    )

    download = str(
        request.args.get(
            "download",
            "false",
        )
    ).lower() in {
        "1",
        "true",
        "yes",
    }

    return send_file(
        BytesIO(result["file_bytes"]),
        mimetype=result["content_type"],
        as_attachment=download,
        download_name=result["filename"],
        max_age=0,
    )