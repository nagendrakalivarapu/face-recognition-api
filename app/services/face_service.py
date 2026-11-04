import os
import cv2
import numpy as np

from pathlib import Path
from threading import RLock

from insightface.app import FaceAnalysis


class FaceService:

    def __init__(self):

        # ========================================================
        # PROJECT DIRECTORIES
        # ========================================================

        self.base_dir = Path(__file__).resolve().parents[2]

        self.embedding_folder = (
                self.base_dir / "embeddings"
        )

        self.upload_folder = (
                self.base_dir / "uploads"
        )

        self.embedding_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.upload_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # ========================================================
        # CONFIGURATION
        # ========================================================

        self.model_name = os.getenv(
            "FACE_MODEL",
            "buffalo_l"
        )

        self.det_size = int(
            os.getenv(
                "FACE_DET_SIZE",
                "640"
            )
        )

        self.ctx_id = int(
            os.getenv(
                "FACE_CTX_ID",
                "0"
            )
        )

        # Recognition threshold
        self.recognition_threshold = float(
            os.getenv(
                "FACE_RECOGNITION_THRESHOLD",
                "0.60"
            )
        )

        # Difference required between best and second-best match
        self.min_margin = float(
            os.getenv(
                "FACE_MIN_MARGIN",
                "0.02"
            )
        )

        # Minimum detector confidence
        self.min_detection_score = float(
            os.getenv(
                "FACE_MIN_DET_SCORE",
                "0.50"
            )
        )

        # Minimum face dimensions
        self.min_face_size = int(
            os.getenv(
                "FACE_MIN_FACE_SIZE",
                "80"
            )
        )

        # Blur threshold
        self.min_blur_score = float(
            os.getenv(
                "FACE_MIN_BLUR_SCORE",
                "45"
            )
        )

        # Brightness range
        self.min_brightness = float(
            os.getenv(
                "FACE_MIN_BRIGHTNESS",
                "35"
            )
        )

        self.max_brightness = float(
            os.getenv(
                "FACE_MAX_BRIGHTNESS",
                "235"
            )
        )

        # Resize very large camera images
        self.max_image_size = int(
            os.getenv(
                "FACE_MAX_IMAGE_SIZE",
                "1600"
            )
        )

        # ========================================================
        # THREAD SAFETY
        # ========================================================

        self.cache_lock = RLock()

        # ========================================================
        # EMBEDDING CACHE
        #
        # Example:
        #
        # {
        #     "12314515": numpy array,
        #     "23CSE001": numpy array
        # }
        # ========================================================

        self.embedding_cache = {}

        # ========================================================
        # LOAD INSIGHTFACE MODEL
        # ========================================================

        print(
            "Loading InsightFace model..."
        )

        self.face_app = FaceAnalysis(
            name=self.model_name
        )

        self.face_app.prepare(
            ctx_id=self.ctx_id,
            det_size=(
                self.det_size,
                self.det_size
            )
        )

        print(
            f"InsightFace model "
            f"'{self.model_name}' loaded successfully."
        )

        # ========================================================
        # LOAD EMBEDDINGS
        # ========================================================

        self.reload_embeddings()

    # ============================================================
    # IMAGE READING
    # ============================================================

    def _read_image(
            self,
            image_path: str
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:
            return None

        height, width = image.shape[:2]

        largest_dimension = max(
            height,
            width
        )

        # Resize very large images
        if largest_dimension > self.max_image_size:

            scale = (
                    self.max_image_size /
                    float(largest_dimension)
            )

            new_width = int(
                width * scale
            )

            new_height = int(
                height * scale
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )

        return image

    # ============================================================
    # BRIGHTNESS
    # ============================================================

    def _calculate_brightness(
            self,
            image
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return float(
            np.mean(gray)
        )

    # ============================================================
    # BLUR
    # ============================================================

    def _calculate_blur_score(
            self,
            image
    ):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

    # ============================================================
    # FACE CROP
    # ============================================================

    def _get_face_crop(
            self,
            image,
            face
    ):

        height, width = image.shape[:2]

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        x1 = max(
            0,
            x1
        )

        y1 = max(
            0,
            y1
        )

        x2 = min(
            width,
            x2
        )

        y2 = min(
            height,
            y2
        )

        if x2 <= x1 or y2 <= y1:
            return None

        return image[
            y1:y2,
            x1:x2
        ]

    # ============================================================
    # DETECT EXACTLY ONE FACE
    # ============================================================

    def _get_single_face(
            self,
            image_path: str
    ):

        image = self._read_image(
            image_path
        )

        if image is None:

            return {
                "success": False,
                "code": "INVALID_IMAGE",
                "message": "Unable to read image."
            }

        # ========================================================
        # BRIGHTNESS CHECK
        # ========================================================

        brightness = (
            self._calculate_brightness(
                image
            )
        )

        if brightness < self.min_brightness:

            return {
                "success": False,
                "code": "LOW_BRIGHTNESS",
                "message": (
                    "Image is too dark. "
                    "Please improve the lighting."
                )
            }

        if brightness > self.max_brightness:

            return {
                "success": False,
                "code": "HIGH_BRIGHTNESS",
                "message": (
                    "Image is too bright. "
                    "Please reduce the lighting."
                )
            }

        # ========================================================
        # FACE DETECTION
        # ========================================================

        faces = self.face_app.get(
            image
        )

        # ========================================================
        # ZERO FACES
        # ========================================================

        if len(faces) == 0:

            return {
                "success": False,
                "code": "NO_FACE",
                "message": (
                    "No face detected. "
                    "Please move into the camera frame."
                )
            }

        # ========================================================
        # MULTIPLE FACES
        #
        # IMPORTANT:
        # Never select largest face automatically.
        # ========================================================

        if len(faces) > 1:

            return {
                "success": False,
                "code": "MULTIPLE_FACES",
                "message": (
                    "Multiple faces detected. "
                    "Only one person should be visible "
                    "in the camera."
                )
            }

        face = faces[0]

        # ========================================================
        # DETECTION SCORE
        # ========================================================

        detection_score = float(
            getattr(
                face,
                "det_score",
                0.0
            )
        )

        if (
                detection_score
                < self.min_detection_score
        ):

            return {
                "success": False,
                "code": "LOW_DETECTION_CONFIDENCE",
                "message": (
                    "Face detection confidence is too low. "
                    "Please move closer to the camera."
                )
            }

        # ========================================================
        # FACE SIZE
        # ========================================================

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        face_width = x2 - x1

        face_height = y2 - y1

        if (
                face_width < self.min_face_size
                or
                face_height < self.min_face_size
        ):

            return {
                "success": False,
                "code": "FACE_TOO_SMALL",
                "message": (
                    "Face is too small. "
                    "Please move closer to the camera."
                )
            }

        # ========================================================
        # FACE CROP
        # ========================================================

        face_crop = self._get_face_crop(
            image,
            face
        )

        if face_crop is None:

            return {
                "success": False,
                "code": "INVALID_FACE_REGION",
                "message": (
                    "Unable to process "
                    "the detected face."
                )
            }

        # ========================================================
        # BLUR CHECK
        # ========================================================

        blur_score = (
            self._calculate_blur_score(
                face_crop
            )
        )

        if (
                blur_score
                < self.min_blur_score
        ):

            return {
                "success": False,
                "code": "BLURRY_IMAGE",
                "message": (
                    "Image is too blurry. "
                    "Please hold the camera steady."
                )
            }

        # ========================================================
        # SUCCESS
        # ========================================================

        return {
            "success": True,
            "face": face,
            "image": image,
            "detectionScore": detection_score,
            "faceWidth": face_width,
            "faceHeight": face_height,
            "blurScore": blur_score,
            "brightness": brightness
        }

    # ============================================================
    # NORMALIZE SINGLE EMBEDDING
    # ============================================================

    @staticmethod
    def _normalize_embedding(
            embedding
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm <= 1e-12:
            return None

        return (
                embedding / norm
        )

    # ============================================================
    # NORMALIZE EMBEDDING MATRIX
    #
    # Supports:
    #
    # (512,)
    #
    # and
    #
    # (N, 512)
    # ============================================================

    @staticmethod
    def _normalize_embedding_matrix(
            embeddings
    ):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Single embedding
        if embeddings.ndim == 1:

            return FaceService._normalize_embedding(
                embeddings
            )

        # Multiple embeddings
        if embeddings.ndim != 2:
            return None

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms = np.maximum(
            norms,
            1e-12
        )

        return (
                embeddings / norms
        )

    # ============================================================
    # GENERATE EMBEDDING
    # ============================================================

    def generate_embedding(
            self,
            image_path: str
    ):

        result = self._get_single_face(
            image_path
        )

        if not result["success"]:
            return result

        face = result["face"]

        # ========================================================
        # PREFER INSIGHTFACE NORMALIZED EMBEDDING
        # ========================================================

        embedding = getattr(
            face,
            "normed_embedding",
            None
        )

        # Fallback
        if embedding is None:

            embedding = getattr(
                face,
                "embedding",
                None
            )

        if embedding is None:

            return {
                "success": False,
                "code": "EMBEDDING_FAILED",
                "message": (
                    "Unable to generate "
                    "face embedding."
                )
            }

        normalized = (
            self._normalize_embedding(
                embedding
            )
        )

        if normalized is None:

            return {
                "success": False,
                "code": "INVALID_EMBEDDING",
                "message": (
                    "Generated face embedding "
                    "is invalid."
                )
            }

        return {
            "success": True,
            "message": (
                "Embedding generated successfully."
            ),
            "embedding": normalized,
            "detectionScore": result[
                "detectionScore"
            ],
            "faceWidth": result[
                "faceWidth"
            ],
            "faceHeight": result[
                "faceHeight"
            ],
            "blurScore": result[
                "blurScore"
            ],
            "brightness": result[
                "brightness"
            ]
        }

    # ============================================================
    # LOAD EMBEDDINGS INTO RAM
    # ============================================================

    def reload_embeddings(
            self
    ):

        new_cache = {}

        if not self.embedding_folder.exists():
            return

        for file in (
                self.embedding_folder.iterdir()
        ):

            if (
                    file.suffix.lower()
                    != ".npy"
            ):
                continue

            student_id = file.stem

            try:

                embedding = np.load(
                    str(file),
                    allow_pickle=False
                )

                normalized = (
                    self._normalize_embedding_matrix(
                        embedding
                    )
                )

                if normalized is None:

                    print(
                        "Skipping invalid "
                        f"embedding: {file.name}"
                    )

                    continue

                new_cache[
                    student_id
                ] = normalized

            except Exception as exc:

                print(
                    f"Failed loading "
                    f"{file.name}: {exc}"
                )

        with self.cache_lock:

            self.embedding_cache = (
                new_cache
            )

        print(
            "Loaded "
            f"{len(new_cache)} student "
            "embeddings into memory."
        )

    # ============================================================
    # ADD EMBEDDING TO CACHE
    # ============================================================

    def add_embedding_to_cache(
            self,
            student_id: str,
            embedding
    ):

        normalized = (
            self._normalize_embedding_matrix(
                embedding
            )
        )

        if normalized is None:

            raise ValueError(
                "Invalid embedding."
            )

        with self.cache_lock:

            self.embedding_cache[
                str(student_id)
            ] = normalized

    # ============================================================
    # SAVE EMBEDDING
    # ============================================================

    def save_embedding(
            self,
            student_id: str,
            embedding
    ):

        normalized = (
            self._normalize_embedding_matrix(
                embedding
            )
        )

        if normalized is None:

            return {
                "success": False,
                "message": (
                    "Invalid embedding."
                )
            }

        # Keep existing file format compatible
        if (
                normalized.ndim == 2
                and
                normalized.shape[0] == 1
        ):

            normalized = normalized[0]

        embedding_path = (
                self.embedding_folder
                / f"{student_id}.npy"
        )

        np.save(
            str(embedding_path),
            normalized
        )

        self.add_embedding_to_cache(
            student_id,
            normalized
        )

        return {
            "success": True,
            "embeddingPath": str(
                embedding_path.relative_to(
                    self.base_dir
                )
            )
        }

    # ============================================================
    # REGISTER FACE
    # ============================================================

    def register_face(
            self,
            student_id: str,
            image_path: str
    ):

        result = (
            self.generate_embedding(
                image_path
            )
        )

        if not result["success"]:
            return result

        embedding = result[
            "embedding"
        ]

        # ========================================================
        # SAVE IMAGE
        # ========================================================

        image_destination = (
                self.upload_folder
                / f"{student_id}.jpg"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            return {
                "success": False,
                "code": "IMAGE_SAVE_FAILED",
                "message": (
                    "Unable to read "
                    "uploaded face image."
                )
            }

        saved = cv2.imwrite(
            str(image_destination),
            image
        )

        if not saved:

            return {
                "success": False,
                "code": "IMAGE_SAVE_FAILED",
                "message": (
                    "Failed to save "
                    "face image."
                )
            }

        # ========================================================
        # SAVE EMBEDDING
        # ========================================================

        save_result = (
            self.save_embedding(
                student_id,
                embedding
            )
        )

        if not save_result["success"]:
            return save_result

        return {
            "success": True,
            "studentId": str(
                student_id
            ),
            "imagePath": str(
                image_destination.relative_to(
                    self.base_dir
                )
            ),
            "embeddingPath": (
                save_result[
                    "embeddingPath"
                ]
            ),
            "message": (
                "Student registered successfully."
            ),
            "quality": {
                "detectionScore": round(
                    float(
                        result[
                            "detectionScore"
                        ]
                    ),
                    4
                ),
                "faceWidth": int(
                    result[
                        "faceWidth"
                    ]
                ),
                "faceHeight": int(
                    result[
                        "faceHeight"
                    ]
                ),
                "blurScore": round(
                    float(
                        result[
                            "blurScore"
                        ]
                    ),
                    2
                ),
                "brightness": round(
                    float(
                        result[
                            "brightness"
                        ]
                    ),
                    2
                )
            }
        }

    # ============================================================
    # RECOGNIZE FACE
    # ============================================================

    def recognize_face(
            self,
            image_path: str
    ):

        result = (
            self.generate_embedding(
                image_path
            )
        )

        if not result["success"]:
            return result

        unknown_embedding = result[
            "embedding"
        ]

        with self.cache_lock:

            if not self.embedding_cache:

                return {
                    "success": False,
                    "code": (
                        "NO_REGISTERED_STUDENTS"
                    ),
                    "message": (
                        "No registered "
                        "students found."
                    )
                }

            best_student = None

            best_score = -1.0

            scores = []

            # ====================================================
            # COMPARE AGAINST CACHE
            # ====================================================

            for (
                    student_id,
                    saved_embeddings
            ) in self.embedding_cache.items():

                # Single reference
                if saved_embeddings.ndim == 1:

                    score = float(
                        np.dot(
                            unknown_embedding,
                            saved_embeddings
                        )
                    )

                # Multiple references
                else:

                    reference_scores = (
                            saved_embeddings
                            @ unknown_embedding
                    )

                    score = float(
                        np.max(
                            reference_scores
                        )
                    )

                scores.append(
                    (
                        student_id,
                        score
                    )
                )

                if score > best_score:

                    best_score = score

                    best_student = (
                        student_id
                    )

        # ========================================================
        # SORT RESULTS
        # ========================================================

        scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        second_best_score = (
            scores[1][1]
            if len(scores) > 1
            else -1.0
        )

        margin = (
            best_score
            - second_best_score
            if len(scores) > 1
            else 1.0
        )

        # ========================================================
        # THRESHOLD
        # ========================================================

        if (
                best_score
                < self.recognition_threshold
        ):

            return {
                "success": False,
                "code": "UNKNOWN_PERSON",
                "message": "Unknown Person.",
                "confidence": round(
                    float(best_score),
                    4
                )
            }

        # ========================================================
        # AMBIGUOUS MATCH
        # ========================================================

        if (
                len(scores) > 1
                and margin < self.min_margin
        ):

            return {
                "success": False,
                "code": "AMBIGUOUS_MATCH",
                "message": (
                    "Face match is ambiguous. "
                    "Please look directly "
                    "at the camera."
                ),
                "confidence": round(
                    float(best_score),
                    4
                )
            }

        # ========================================================
        # SUCCESS
        # ========================================================

        return {
            "success": True,
            "studentId": best_student,
            "confidence": round(
                float(best_score),
                4
            ),
            "message": (
                "Face recognized successfully."
            )
        }

    # ============================================================
    # DETECT FACE ONLY
    # ============================================================

    def detect_face(
            self,
            image_path: str
    ):

        result = (
            self._get_single_face(
                image_path
            )
        )

        if not result["success"]:
            return result

        return {
            "success": True,
            "message": (
                "Exactly one face detected."
            ),
            "faceCount": 1,
            "quality": {
                "detectionScore": round(
                    float(
                        result[
                            "detectionScore"
                        ]
                    ),
                    4
                ),
                "faceWidth": int(
                    result[
                        "faceWidth"
                    ]
                ),
                "faceHeight": int(
                    result[
                        "faceHeight"
                    ]
                ),
                "blurScore": round(
                    float(
                        result[
                            "blurScore"
                        ]
                    ),
                    2
                ),
                "brightness": round(
                    float(
                        result[
                            "brightness"
                        ]
                    ),
                    2
                )
            }
        }


# ================================================================
# GLOBAL FACE SERVICE
#
# InsightFace model is loaded only once.
# ================================================================

face_service = FaceService()