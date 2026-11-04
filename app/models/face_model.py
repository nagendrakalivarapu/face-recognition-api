from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# FACE QUALITY
# ============================================================

class FaceQuality(BaseModel):
    detectionScore: float = Field(
        ...,
        description="Face detection confidence score"
    )

    faceWidth: int = Field(
        ...,
        description="Detected face width in pixels"
    )

    faceHeight: int = Field(
        ...,
        description="Detected face height in pixels"
    )

    blurScore: float = Field(
        ...,
        description="Image sharpness score"
    )

    brightness: float = Field(
        ...,
        description="Average image brightness"
    )


# ============================================================
# FACE DETECTION RESPONSE
# ============================================================

class FaceDetectionResponse(BaseModel):
    success: bool

    message: str

    code: Optional[str] = None

    faceCount: Optional[int] = None

    quality: Optional[FaceQuality] = None


# ============================================================
# FACE RECOGNITION RESPONSE
# ============================================================

class FaceRecognitionResponse(BaseModel):
    success: bool

    message: str

    code: Optional[str] = None

    studentId: Optional[str] = None

    confidence: Optional[float] = None


# ============================================================
# FACE REGISTRATION RESPONSE
# ============================================================

class FaceRegistrationResponse(BaseModel):
    success: bool

    message: str

    studentId: Optional[str] = None

    imagePath: Optional[str] = None

    embeddingPath: Optional[str] = None

    quality: Optional[FaceQuality] = None


# ============================================================
# GENERIC FACE ERROR RESPONSE
# ============================================================

class FaceErrorResponse(BaseModel):
    success: bool = False

    message: str

    code: Optional[str] = None