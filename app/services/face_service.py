from insightface.app import FaceAnalysis
import cv2
import os
import numpy as np

from app.utils.cosine_similarity import cosine_similarity


class FaceService:

    def __init__(self):
        """
        Initialize InsightFace model.
        """
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def _get_single_face(self, image_path: str):
        """
        Internal helper method to detect exactly one face.
        """

        image = cv2.imread(image_path)

        if image is None:
            return {
                "success": False,
                "message": "Unable to read image."
            }

        faces = self.face_app.get(image)

        if len(faces) == 0:
            return {
                "success": False,
                "message": "No face detected."
            }

        if len(faces) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected."
            }

        return {
            "success": True,
            "face": faces[0]
        }

    def detect_face(self, image_path: str):
        """
        Check whether exactly one face exists.
        """

        result = self._get_single_face(image_path)

        if not result["success"]:
            return result

        return {
            "success": True,
            "message": "Exactly one face detected."
        }

    def generate_embedding(self, image_path: str):
        """
        Generate a 512-dimensional face embedding.
        """

        result = self._get_single_face(image_path)

        if not result["success"]:
            return result

        embedding = result["face"].embedding

        return {
            "success": True,
            "message": "Embedding generated successfully.",
            "embedding": embedding
        }

    def recognize_face(self, image_path: str):
        """
        Compare uploaded face with stored embeddings.
        """

        result = self.generate_embedding(image_path)

        if not result["success"]:
            return result

        unknown_embedding = result["embedding"]

        embedding_folder = "embeddings"

        if not os.path.exists(embedding_folder):
            return {
                "success": False,
                "message": "Embeddings folder not found."
            }

        best_match = None
        highest_score = -1.0

        for file in os.listdir(embedding_folder):

            if not file.endswith(".npy"):
                continue

            student_id = file.replace(".npy", "")

            saved_embedding = np.load(
                os.path.join(embedding_folder, file)
            )

            score = cosine_similarity(
                unknown_embedding,
                saved_embedding
            )

            if score > highest_score:
                highest_score = score
                best_match = student_id

        if best_match is None:
            return {
                "success": False,
                "message": "No registered students found."
            }

        THRESHOLD = 0.60

        if highest_score >= THRESHOLD:
            return {
                "success": True,
                "studentId": best_match,
                "confidence": round(float(highest_score), 4),
                "message": "Face recognized successfully."
            }

        return {
            "success": False,
            "message": "Unknown Person",
            "confidence": round(float(highest_score), 4)
        }


# Singleton instance
face_service = FaceService()