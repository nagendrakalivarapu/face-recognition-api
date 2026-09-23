from fastapi import APIRouter, UploadFile, File, Form
from app.services.face_service import face_service
import os
import shutil
import numpy as np

router = APIRouter(
    prefix="/face",
    tags=["Face Recognition"]
)

UPLOAD_FOLDER = "uploads"
EMBEDDING_FOLDER = "embeddings"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "Face Recognition API is Working"
    }


@router.post("/register")
async def register_face(
    student_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Save uploaded image
        filename = f"{student_id}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Generate face embedding
        result = face_service.generate_embedding(image_path)

        # Face validation failed
        if not result["success"]:
            if os.path.exists(image_path):
                os.remove(image_path)

            return {
                "success": False,
                "message": result["message"]
            }

        # Save embedding
        embedding_path = os.path.join(
            EMBEDDING_FOLDER,
            f"{student_id}.npy"
        )

        np.save(embedding_path, result["embedding"])

        return {
            "success": True,
            "studentId": student_id,
            "imagePath": image_path,
            "embeddingPath": embedding_path,
            "message": "Student registered successfully."
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
    
@router.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...)
):

    filename = "temp.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = face_service.recognize_face(filepath)

    if os.path.exists(filepath):
        os.remove(filepath)

    return result