import logging
import os
import re
import secrets
import tempfile
from io import BytesIO
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.models.face_model import (
    FaceDetectionResponse,
    FaceRecognitionResponse,
    FaceRegistrationResponse,
)
from app.services.face_service import face_service

router = APIRouter(prefix="/face", tags=["Face Recognition"])

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 16_000_000

ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}


# ============================================================
# INTERNAL API AUTHENTICATION
# ============================================================


async def verify_internal_api_key(
    x_face_api_key: str | None = Header(
        default=None,
        alias="X-Face-API-Key",
    ),
):
    expected_key = os.getenv("FACE_API_KEY")

    if not expected_key or len(expected_key) < 32:
        logger.error("Face API authentication is not configured.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face service authentication is unavailable.",
        )

    if x_face_api_key is None or not secrets.compare_digest(
        x_face_api_key, expected_key
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized face API request.",
        )


# ============================================================
# HEALTH
# ============================================================


@router.get("/health")
async def health():

    return {"status": "healthy", "message": "Face Recognition API is Working"}


# ============================================================
# VALIDATE AND STORE UPLOAD
# ============================================================


async def save_validated_image(image: UploadFile) -> str:

    if image is None:
        raise HTTPException(status_code=400, detail="Image file is required.")

    content = await image.read(MAX_IMAGE_SIZE + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty.")

    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image must not exceed 5 MB.")

    try:
        image_data = BytesIO(content)

        with Image.open(image_data) as img:
            detected_format = img.format

            if detected_format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only JPEG and PNG images are allowed.",
                )

            width, height = img.size

            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_WIDTH
                or height > MAX_IMAGE_HEIGHT
                or width * height > MAX_IMAGE_PIXELS
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Image dimensions exceed allowed limits.",
                )

            img.verify()

        image_data.seek(0)
        with Image.open(image_data) as img:
            img.load()

            if img.format != detected_format:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image content.",
                )

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.")

    suffix = ALLOWED_FORMATS[detected_format]

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="face_", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)

        return temp_path

    except OSError:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        logger.exception("Failed to create temporary face image.")

        raise HTTPException(status_code=500, detail="Unable to process image.")


# ============================================================
# TEMPORARY FILE CLEANUP
# ============================================================


def cleanup_temp_file(path: str | None):

    if not path:
        return

    try:
        os.remove(path)

    except FileNotFoundError:
        pass

    except OSError:
        logger.exception("Temporary face image cleanup failed.")


# ============================================================
# STUDENT ID VALIDATION
# ============================================================


def validate_student_id(student_id: str):

    student_id = student_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", student_id) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid student ID.",
        )

    return student_id


# ============================================================
# REGISTER FACE
# ============================================================


@router.post(
    "/register",
    response_model=FaceRegistrationResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def register_face(
    student_id: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
):
    temp_path = None

    try:
        student_id = validate_student_id(student_id)
        temp_path = await save_validated_image(image)

        result = await run_in_threadpool(
            face_service.register_face, student_id, temp_path
        )

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("Face registration failed.")

        return {"success": False, "message": "Face registration failed."}

    finally:
        cleanup_temp_file(temp_path)

        await image.close()


# ============================================================
# RECOGNIZE FACE
# ============================================================


@router.post(
    "/recognize",
    response_model=FaceRecognitionResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def recognize_face(
    image: Annotated[UploadFile, File()],
):

    temp_path = None

    try:
        temp_path = await save_validated_image(image)

        result = await run_in_threadpool(face_service.recognize_face, temp_path)

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("Face recognition failed.")

        return {"success": False, "message": "Face recognition failed."}

    finally:
        cleanup_temp_file(temp_path)

        await image.close()


# ============================================================
# DETECT FACE
# ============================================================


@router.post(
    "/detect",
    response_model=FaceDetectionResponse,
    dependencies=[Depends(verify_internal_api_key)],
)
async def detect_face(
    image: Annotated[UploadFile, File()],
):

    temp_path = None

    try:
        temp_path = await save_validated_image(image)

        result = await run_in_threadpool(face_service.detect_face, temp_path)

        return result

    except HTTPException:
        raise

    except Exception:
        logger.exception("Face detection failed.")

        return {"success": False, "message": "Face detection failed."}

    finally:
        cleanup_temp_file(temp_path)

        await image.close()
