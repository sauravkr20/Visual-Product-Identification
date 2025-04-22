from fastapi import APIRouter, UploadFile, File
from app.controllers.add_controller import AddController

router = APIRouter()

add_controller: AddController = None  # Initialized in main.py

@router.post("/add/")
async def add_image(file: UploadFile = File(...), image_path: str = None):
    return await add_controller.add_image(file, image_path)
