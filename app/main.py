from fastapi import FastAPI
from app.routes.face import router as face_router

app = FastAPI(
    title="Smart Attendance Face Recognition API",
    version="1.0.0"
)

app.include_router(face_router)


@app.get("/")
async def home():
    return {
        "status": "running",
        "message": "Face Recognition API is Ready!"
    }