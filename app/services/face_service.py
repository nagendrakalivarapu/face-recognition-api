import logging
import math
import os
import re
import tempfile
from pathlib import Path
from threading import RLock

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceService:
    STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

    MAX_EMBEDDING_FILE_SIZE = 1024 * 1024
    MAX_REFERENCE_EMBEDDINGS = 10

    def __init__(self):

        # ========================================================
        # PROJECT DIRECTORIES
        # ========================================================

        self.base_dir = Path(__file__).resolve().parents[2]

        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_folder = self.base_dir / "embeddings"
        self.upload_folder = self.base_dir / "uploads"

        self.embedding_folder.mkdir(parents=True, exist_ok=True)
        self.upload_folder.mkdir(parents=True, exist_ok=True)

        self.embedding_folder = self.embedding_folder.resolve()
        self.upload_folder = self.upload_folder.resolve()

        # ========================================================
        # CONFIGURATION
        # ========================================================

        self.model_name = os.getenv("FACE_MODEL", "buffalo_l").strip()

        self.det_size = int(os.getenv("FACE_DET_SIZE", "640"))
        self.ctx_id = int(os.getenv("FACE_CTX_ID", "0"))

        self.recognition_threshold = float(
            os.getenv("FACE_RECOGNITION_THRESHOLD", "0.60")
        )

        self.min_margin = float(os.getenv("FACE_MIN_MARGIN", "0.02"))

        self.min_detection_score = float(os.getenv("FACE_MIN_DET_SCORE", "0.50"))

        self.min_face_size = int(os.getenv("FACE_MIN_FACE_SIZE", "80"))

        self.min_blur_score = float(os.getenv("FACE_MIN_BLUR_SCORE", "45"))

        self.min_brightness = float(os.getenv("FACE_MIN_BRIGHTNESS", "35"))

        self.max_brightness = float(os.getenv("FACE_MAX_BRIGHTNESS", "235"))

        self.max_image_size = int(os.getenv("FACE_MAX_IMAGE_SIZE", "1600"))

        # InsightFace buffalo_l uses 512-dimensional embeddings.
        self.embedding_dimension = int(os.getenv("FACE_EMBEDDING_DIM", "512"))

        self._validate_configuration()

        # ========================================================
        # THREAD SAFETY
        # ========================================================

        self.cache_lock = RLock()
        self.inference_lock = RLock()
        self.registration_lock = RLock()

        self.embedding_cache = {}

        # ========================================================
        # LOAD INSIGHTFACE MODEL
        # ========================================================

        logger.info("Loading InsightFace model: %s", self.model_name)

        self.face_app = FaceAnalysis(name=self.model_name)

        self.face_app.prepare(
            ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size)
        )

        logger.info("InsightFace model loaded successfully.")

        # ========================================================
        # LOAD EXISTING EMBEDDINGS
        # ========================================================

        self.reload_embeddings()

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_configuration(self):

        if not self.model_name:
            raise ValueError("FACE_MODEL must not be empty.")

        if self.det_size <= 0:
            raise ValueError("FACE_DET_SIZE must be positive.")

        if self.embedding_dimension <= 0:
            raise ValueError("FACE_EMBEDDING_DIM must be positive.")

        if self.min_face_size <= 0:
            raise ValueError("FACE_MIN_FACE_SIZE must be positive.")

        if self.max_image_size <= 0:
            raise ValueError("FACE_MAX_IMAGE_SIZE must be positive.")

        thresholds = (
            self.recognition_threshold,
            self.min_margin,
            self.min_detection_score,
            self.min_blur_score,
            self.min_brightness,
            self.max_brightness,
        )
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("Face recognition thresholds must be finite.")

        if not 0.0 <= self.recognition_threshold <= 1.0:
            raise ValueError("Invalid face recognition threshold.")

        if not 0.0 <= self.min_margin <= 2.0:
            raise ValueError("Invalid face recognition margin.")

        if not 0.0 <= self.min_detection_score <= 1.0:
            raise ValueError("Invalid face detection threshold.")

        if self.min_blur_score < 0.0:
            raise ValueError("FACE_MIN_BLUR_SCORE must not be negative.")

        if not 0.0 <= self.min_brightness < self.max_brightness <= 255.0:
            raise ValueError("Invalid brightness configuration.")

    # ============================================================
    # STUDENT ID VALIDATION
    # ============================================================

    @classmethod
    def validate_student_id(cls, student_id: str) -> str:

        if not isinstance(student_id, str):
            raise TypeError("Invalid student ID.")

        student_id = student_id.strip()

        if not cls.STUDENT_ID_PATTERN.fullmatch(student_id):
            raise ValueError("Invalid student ID.")

        return student_id

    # ============================================================
    # SAFE STORAGE PATHS
    # ============================================================

    def _get_embedding_path(self, student_id: str) -> Path:

        student_id = self.validate_student_id(student_id)

        path = self.embedding_folder / f"{student_id}.npy"

        if path.parent != self.embedding_folder:
            raise ValueError("Invalid embedding path.")

        return path

    def _get_image_path(self, student_id: str) -> Path:

        student_id = self.validate_student_id(student_id)

        path = self.upload_folder / f"{student_id}.jpg"

        if path.parent != self.upload_folder:
            raise ValueError("Invalid image path.")

        return path

    # ============================================================
    # IMAGE READING
    # ============================================================

    def _read_image(self, image_path: str):

        path = Path(image_path)

        if not path.is_file():
            return None

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            return None

        if image.ndim != 3 or image.shape[2] != 3:
            return None

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return None

        largest_dimension = max(height, width)

        if largest_dimension > self.max_image_size:
            scale = self.max_image_size / float(largest_dimension)

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        return image

    # ============================================================
    # BRIGHTNESS
    # ============================================================

    def _calculate_brightness(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return float(np.mean(gray))

    # ============================================================
    # BLUR
    # ============================================================

    def _calculate_blur_score(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ============================================================
    # FACE CROP
    # ============================================================

    def _get_face_crop(self, image, face):

        height, width = image.shape[:2]

        bbox = np.asarray(face.bbox, dtype=np.float32)

        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            return None

        x1, y1, x2, y2 = bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    # ============================================================
    # DETECT EXACTLY ONE FACE
    # ============================================================

    def _get_single_face(self, image_path: str):

        image = self._read_image(image_path)

        if image is None:
            return {
                "success": False,
                "code": "INVALID_IMAGE",
                "message": "Unable to read image.",
            }

        brightness = self._calculate_brightness(image)

        if not np.isfinite(brightness):
            return {
                "success": False,
                "code": "INVALID_IMAGE",
                "message": "Unable to process image.",
            }

        if brightness < self.min_brightness:
            return {
                "success": False,
                "code": "LOW_BRIGHTNESS",
                "message": ("Image is too dark. Please improve the lighting."),
            }

        if brightness > self.max_brightness:
            return {
                "success": False,
                "code": "HIGH_BRIGHTNESS",
                "message": ("Image is too bright. Please reduce the lighting."),
            }

        # Protect model inference against concurrent access.
        with self.inference_lock:
            faces = self.face_app.get(image)

        if len(faces) == 0:
            return {
                "success": False,
                "code": "NO_FACE",
                "message": ("No face detected. Please move into the camera frame."),
            }

        if len(faces) > 1:
            return {
                "success": False,
                "code": "MULTIPLE_FACES",
                "message": (
                    "Multiple faces detected. "
                    "Only one person should be visible "
                    "in the camera."
                ),
            }

        face = faces[0]

        detection_score = float(getattr(face, "det_score", 0.0))

        if (
            not np.isfinite(detection_score)
            or detection_score < self.min_detection_score
        ):
            return {
                "success": False,
                "code": "LOW_DETECTION_CONFIDENCE",
                "message": (
                    "Face detection confidence is too low. "
                    "Please move closer to the camera."
                ),
            }

        bbox = np.asarray(face.bbox, dtype=np.float32)

        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            return {
                "success": False,
                "code": "INVALID_FACE_REGION",
                "message": "Unable to process the detected face.",
            }

        x1, y1, x2, y2 = bbox.astype(int)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < self.min_face_size or face_height < self.min_face_size:
            return {
                "success": False,
                "code": "FACE_TOO_SMALL",
                "message": ("Face is too small. Please move closer to the camera."),
            }

        face_crop = self._get_face_crop(image, face)

        if face_crop is None:
            return {
                "success": False,
                "code": "INVALID_FACE_REGION",
                "message": "Unable to process the detected face.",
            }

        blur_score = self._calculate_blur_score(face_crop)

        if not np.isfinite(blur_score) or blur_score < self.min_blur_score:
            return {
                "success": False,
                "code": "BLURRY_IMAGE",
                "message": ("Image is too blurry. Please hold the camera steady."),
            }

        return {
            "success": True,
            "face": face,
            "image": image,
            "detectionScore": detection_score,
            "faceWidth": face_width,
            "faceHeight": face_height,
            "blurScore": blur_score,
            "brightness": brightness,
        }

    # ============================================================
    # NORMALIZE SINGLE EMBEDDING
    # ============================================================

    @staticmethod
    def _normalize_embedding(embedding):

        try:
            embedding = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if embedding.ndim != 1:
            return None

        if not np.all(np.isfinite(embedding)):
            return None

        norm = float(np.linalg.norm(embedding))

        if not np.isfinite(norm) or norm <= 1e-12:
            return None

        return embedding / norm

    # ============================================================
    # NORMALIZE EMBEDDING MATRIX
    # ============================================================

    def _normalize_embedding_matrix(self, embeddings):

        try:
            embeddings = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError):
            return None

        if not np.all(np.isfinite(embeddings)):
            return None

        if embeddings.ndim == 1:
            if embeddings.shape[0] != self.embedding_dimension:
                return None

            return self._normalize_embedding(embeddings)

        if embeddings.ndim != 2:
            return None

        if embeddings.shape[1] != self.embedding_dimension:
            return None

        if (
            embeddings.shape[0] < 1
            or embeddings.shape[0] > self.MAX_REFERENCE_EMBEDDINGS
        ):
            return None

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        if not np.all(np.isfinite(norms)):
            return None

        if np.any(norms <= 1e-12):
            return None

        return embeddings / norms

    # ============================================================
    # GENERATE EMBEDDING
    # ============================================================

    def generate_embedding(self, image_path: str):

        result = self._get_single_face(image_path)

        if not result["success"]:
            return result

        face = result["face"]

        embedding = getattr(face, "normed_embedding", None)

        if embedding is None:
            embedding = getattr(face, "embedding", None)

        if embedding is None:
            return {
                "success": False,
                "code": "EMBEDDING_FAILED",
                "message": "Unable to generate face embedding.",
            }

        normalized = self._normalize_embedding(embedding)

        if normalized is None or normalized.shape[0] != self.embedding_dimension:
            return {
                "success": False,
                "code": "INVALID_EMBEDDING",
                "message": "Generated face embedding is invalid.",
            }

        return {
            "success": True,
            "message": "Embedding generated successfully.",
            # Required by register_face()
            "image": result["image"],
            "embedding": normalized,
            "detectionScore": result["detectionScore"],
            "faceWidth": result["faceWidth"],
            "faceHeight": result["faceHeight"],
            "blurScore": result["blurScore"],
            "brightness": result["brightness"],
        }

    # ============================================================
    # LOAD EMBEDDINGS INTO RAM
    # ============================================================

    def reload_embeddings(self):

        new_cache = {}

        for file in self.embedding_folder.iterdir():
            if file.suffix.lower() != ".npy":
                continue

            try:
                student_id = self.validate_student_id(file.stem)

                # Reject symbolic links.
                if file.is_symlink() or not file.is_file():
                    logger.warning("Skipping unsafe embedding file: %s", file.name)
                    continue

                if file.stat().st_size > self.MAX_EMBEDDING_FILE_SIZE:
                    logger.warning("Skipping oversized embedding file: %s", file.name)
                    continue

                embedding = np.load(str(file), allow_pickle=False)

                normalized = self._normalize_embedding_matrix(embedding)

                if normalized is None:
                    logger.warning("Skipping invalid embedding file: %s", file.name)
                    continue

                # Keep a private copy in memory.
                new_cache[student_id] = normalized.copy()

            except Exception:
                logger.exception("Failed loading embedding file: %s", file.name)

        with self.cache_lock:
            self.embedding_cache = new_cache

        logger.info("Loaded %d student embeddings into memory.", len(new_cache))

    # ============================================================
    # ADD EMBEDDING TO CACHE
    # ============================================================

    def add_embedding_to_cache(self, student_id: str, embedding):

        student_id = self.validate_student_id(student_id)

        normalized = self._normalize_embedding_matrix(embedding)

        if normalized is None:
            raise ValueError("Invalid embedding.")

        with self.cache_lock:
            self.embedding_cache[student_id] = normalized.copy()

    # ============================================================
    # SAVE EMBEDDING
    # ============================================================

    def save_embedding(self, student_id: str, embedding):

        student_id = self.validate_student_id(student_id)

        normalized = self._normalize_embedding_matrix(embedding)

        if normalized is None:
            return {"success": False, "message": "Invalid embedding."}

        if normalized.ndim == 2 and normalized.shape[0] == 1:
            normalized = normalized[0]

        embedding_path = self._get_embedding_path(student_id)

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npy",
                prefix=".embedding_",
                dir=self.embedding_folder,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

                np.save(temp_file, normalized, allow_pickle=False)

                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Atomic replacement of a single file.
            os.replace(temp_path, embedding_path)

            self.add_embedding_to_cache(student_id, normalized)

            return {
                "success": True,
                "embeddingPath": str(embedding_path.relative_to(self.base_dir)),
            }

        except (OSError, ValueError):
            logger.exception("Failed to save face embedding.")

            return {"success": False, "message": "Unable to save face embedding."}

        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Unable to remove temporary embedding file.")

    # ============================================================
    # REGISTER FACE
    # ============================================================

    def register_face(self, student_id: str, image_path: str):

        try:
            student_id = self.validate_student_id(student_id)
        except (TypeError, ValueError):
            return {"success": False, "message": "Invalid student ID."}

        with self.registration_lock:
            image_destination = self._get_image_path(student_id)
            embedding_destination = self._get_embedding_path(student_id)

            # Prevent silent overwriting of existing registrations.
            with self.cache_lock:
                already_registered = (
                    student_id in self.embedding_cache
                    or image_destination.exists()
                    or embedding_destination.exists()
                )

            if already_registered:
                return {
                    "success": False,
                    "studentId": student_id,
                    "message": "Face is already registered for this student.",
                }

            result = self.generate_embedding(image_path)

            if not result["success"]:
                return result

            image = result["image"]

            success, encoded_image = cv2.imencode(".jpg", image)

            if not success:
                return {
                    "success": False,
                    "code": "IMAGE_SAVE_FAILED",
                    "message": "Failed to encode face image.",
                }

            embedding = result["embedding"]

            image_temp_path = None
            embedding_temp_path = None

            image_committed = False
            embedding_committed = False

            try:
                # Stage image in the same directory.
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".jpg",
                    prefix=".face_",
                    dir=self.upload_folder,
                    delete=False,
                ) as temp_file:
                    image_temp_path = Path(temp_file.name)

                    temp_file.write(encoded_image.tobytes())
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                # Stage embedding in the same directory.
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".npy",
                    prefix=".embedding_",
                    dir=self.embedding_folder,
                    delete=False,
                ) as temp_file:
                    embedding_temp_path = Path(temp_file.name)

                    np.save(temp_file, embedding, allow_pickle=False)

                    temp_file.flush()
                    os.fsync(temp_file.fileno())

                # Recheck immediately before committing.
                if image_destination.exists() or embedding_destination.exists():
                    return {
                        "success": False,
                        "studentId": student_id,
                        "message": "Face is already registered for this student.",
                    }

                # Commit both files.
                os.replace(image_temp_path, image_destination)
                image_temp_path = None
                image_committed = True

                os.replace(embedding_temp_path, embedding_destination)
                embedding_temp_path = None
                embedding_committed = True

                self.add_embedding_to_cache(student_id, embedding)

                return {
                    "success": True,
                    "studentId": student_id,
                    "imagePath": str(image_destination.relative_to(self.base_dir)),
                    "embeddingPath": str(
                        embedding_destination.relative_to(self.base_dir)
                    ),
                    "message": "Student registered successfully.",
                    "quality": {
                        "detectionScore": round(float(result["detectionScore"]), 4),
                        "faceWidth": int(result["faceWidth"]),
                        "faceHeight": int(result["faceHeight"]),
                        "blurScore": round(float(result["blurScore"]), 2),
                        "brightness": round(float(result["brightness"]), 2),
                    },
                }

            except Exception:
                logger.exception("Face registration storage failed.")

                # Roll back only files created by this registration.
                if image_committed:
                    image_destination.unlink(missing_ok=True)

                if embedding_committed:
                    embedding_destination.unlink(missing_ok=True)

                with self.cache_lock:
                    self.embedding_cache.pop(student_id, None)

                return {
                    "success": False,
                    "code": "REGISTRATION_FAILED",
                    "studentId": student_id,
                    "message": "Face registration failed.",
                }

            finally:
                for temp_path in (image_temp_path, embedding_temp_path):
                    if temp_path is not None:
                        try:
                            temp_path.unlink(missing_ok=True)
                        except OSError:
                            logger.warning(
                                "Unable to remove temporary registration file."
                            )

    # ============================================================
    # RECOGNIZE FACE
    # ============================================================

    def recognize_face(self, image_path: str):

        result = self.generate_embedding(image_path)

        if not result["success"]:
            return result

        unknown_embedding = result["embedding"]

        with self.cache_lock:
            references = tuple(self.embedding_cache.items())

        if not references:
            return {
                "success": False,
                "code": "NO_REGISTERED_STUDENTS",
                "message": "No registered students found.",
            }

        best_student = None
        best_score = -np.inf
        second_best_score = -1.0
        score_count = 0

        for student_id, saved_embeddings in references:
            # Cache entries are normalized when loaded or written.
            if saved_embeddings.ndim == 1:
                score = float(np.dot(unknown_embedding, saved_embeddings))

            else:
                reference_scores = saved_embeddings @ unknown_embedding
                score = float(np.max(reference_scores))

            if not np.isfinite(score):
                continue

            score_count += 1

            if score > best_score:
                if best_student is not None:
                    second_best_score = best_score

                best_student = student_id
                best_score = score

            elif score > second_best_score:
                second_best_score = score

        if best_student is None:
            return {
                "success": False,
                "code": "NO_REGISTERED_STUDENTS",
                "message": "No valid registered faces found.",
            }

        margin = best_score - second_best_score

        if best_score < self.recognition_threshold:
            return {
                "success": False,
                "code": "UNKNOWN_PERSON",
                "message": "Unknown Person.",
                "confidence": round(float(best_score), 4),
            }

        if score_count > 1 and margin < self.min_margin:
            return {
                "success": False,
                "code": "AMBIGUOUS_MATCH",
                "message": (
                    "Face match is ambiguous. Please look directly at the camera."
                ),
                "confidence": round(float(best_score), 4),
            }

        return {
            "success": True,
            "studentId": best_student,
            "confidence": round(float(best_score), 4),
            "message": "Face recognized successfully.",
        }

    # ============================================================
    # DETECT FACE ONLY
    # ============================================================

    def detect_face(self, image_path: str):

        result = self._get_single_face(image_path)

        if not result["success"]:
            return result

        return {
            "success": True,
            "message": "Exactly one face detected.",
            "faceCount": 1,
            "quality": {
                "detectionScore": round(float(result["detectionScore"]), 4),
                "faceWidth": int(result["faceWidth"]),
                "faceHeight": int(result["faceHeight"]),
                "blurScore": round(float(result["blurScore"]), 2),
                "brightness": round(float(result["brightness"]), 2),
            },
        }


# ================================================================
# GLOBAL FACE SERVICE
# ================================================================

face_service = FaceService()
