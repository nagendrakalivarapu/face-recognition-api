from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from app.routes.face import (
    router as face_router
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=(
        "Smart Attendance "
        "Face Recognition API"
    ),
    description=(
        "Face recognition service "
        "for Smart Attendance System"
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    face_router
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def home():

    return {
        "status": "running",
        "message": (
            "Face Recognition API is Ready!"
        )
    }