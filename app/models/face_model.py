from pydantic import BaseModel, Field

# ============================================================
# FACE QUALITY
# ============================================================


class FaceQuality(BaseModel):
    detectionScore: float = Field(..., description="Face detection confidence score")

    faceWidth: int = Field(..., description="Detected face width in pixels")

    faceHeight: int = Field(..., description="Detected face height in pixels")

    blurScore: float = Field(..., description="Image sharpness score")

    brightness: float = Field(..., description="Average image brightness")


# ============================================================
# FACE DETECTION RESPONSE
# ============================================================


class FaceDetectionResponse(BaseModel):
    success: bool

    message: str

    code: str | None = None

    faceCount: int | None = None

    quality: FaceQuality | None = None


# ============================================================
# FACE RECOGNITION RESPONSE
# ============================================================


class FaceRecognitionResponse(BaseModel):
    success: bool

    message: str

    code: str | None = None

    studentId: str | None = None

    confidence: float | None = None


# ============================================================
# FACE REGISTRATION RESPONSE
# ============================================================


class FaceRegistrationResponse(BaseModel):
    success: bool

    message: str

    studentId: str | None = None

    imagePath: str | None = None

    embeddingPath: str | None = None

    quality: FaceQuality | None = None


# ============================================================
# GENERIC FACE ERROR RESPONSE
# ============================================================


class FaceErrorResponse(BaseModel):
    success: bool = False

    message: str

    code: str | None = None
