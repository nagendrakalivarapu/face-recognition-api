import os
import tempfile

from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form
)

from app.models.face_model import (
    FaceDetectionResponse,
    FaceRecognitionResponse,
    FaceRegistrationResponse
)

from app.services.face_service import (
    face_service
)


router = APIRouter(
    prefix="/face",
    tags=["Face Recognition"]
)


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def health():

    return {
        "status": "healthy",
        "message": (
            "Face Recognition API is Working"
        )
    }


# ============================================================
# REGISTER FACE
# ============================================================

@router.post(
    "/register",
    response_model=FaceRegistrationResponse
)
async def register_face(
        student_id: str = Form(...),
        image: UploadFile = File(...)
):

    if not student_id.strip():

        return {
            "success": False,
            "message": (
                "student_id is required."
            )
        }

    if image is None:

        return {
            "success": False,
            "message": (
                "Image is required."
            )
        }

    temp_path = None

    try:

        # ====================================================
        # UNIQUE TEMP FILE
        # ====================================================

        suffix = Path(
            image.filename or ".jpg"
        ).suffix

        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            while True:

                chunk = await image.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temp_file.write(
                    chunk
                )

        # ====================================================
        # REGISTER
        # ====================================================

        result = (
            face_service.register_face(
                student_id.strip(),
                temp_path
            )
        )

        return result

    except Exception as exc:

        print(
            f"Face registration error: {exc}"
        )

        return {
            "success": False,
            "message": (
                "Face registration failed."
            )
        }

    finally:

        if (
                temp_path
                and
                os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass


# ============================================================
# RECOGNIZE FACE
# ============================================================

@router.post(
    "/recognize",
    response_model=FaceRecognitionResponse
)
async def recognize_face(
        image: UploadFile = File(...)
):

    if image is None:

        return {
            "success": False,
            "message": (
                "Image is required."
            )
        }

    temp_path = None

    try:

        # ====================================================
        # UNIQUE TEMP FILE
        # ====================================================

        suffix = Path(
            image.filename or ".jpg"
        ).suffix

        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            while True:

                chunk = await image.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temp_file.write(
                    chunk
                )

        # ====================================================
        # RECOGNIZE
        # ====================================================

        result = (
            face_service.recognize_face(
                temp_path
            )
        )

        return result

    except Exception as exc:

        print(
            f"Face recognition error: {exc}"
        )

        return {
            "success": False,
            "message": (
                "Face recognition failed."
            )
        }

    finally:

        if (
                temp_path
                and
                os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass


# ============================================================
# DETECT FACE
# ============================================================

@router.post(
    "/detect",
    response_model=FaceDetectionResponse
)
async def detect_face(
        image: UploadFile = File(...)
):

    if image is None:

        return {
            "success": False,
            "message": (
                "Image is required."
            )
        }

    temp_path = None

    try:

        # ====================================================
        # UNIQUE TEMP FILE
        # ====================================================

        suffix = Path(
            image.filename or ".jpg"
        ).suffix

        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            while True:

                chunk = await image.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temp_file.write(
                    chunk
                )

        # ====================================================
        # DETECT
        # ====================================================

        result = (
            face_service.detect_face(
                temp_path
            )
        )

        return result

    except Exception as exc:

        print(
            f"Face detection error: {exc}"
        )

        return {
            "success": False,
            "message": (
                "Face detection failed."
            )
        }

    finally:

        if (
                temp_path
                and
                os.path.exists(temp_path)
        ):

            try:
                os.remove(
                    temp_path
                )
            except Exception:
                pass