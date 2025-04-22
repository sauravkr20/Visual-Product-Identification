from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

from fastapi import FastAPI

from app.config import SHOE_IMAGES_FOLDER, FAISS_INDEX_PATH, IMAGE_PATHS_JSON
from app.model import extract_embedding
from app.search import load_index, load_image_paths, save_index, save_image_paths, search

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = load_index(FAISS_INDEX_PATH)
image_paths = load_image_paths(IMAGE_PATHS_JSON)

@app.post("/search/")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    emb = extract_embedding(image)
    indices, scores = search(index, emb, top_k)
    results = [{"image_path": image_paths[i], "score": scores[idx]} for idx, i in enumerate(indices)]
    return {"results": results}

@app.post("/add/")
async def add_image(file: UploadFile = File(...), image_path: str = None):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    emb = extract_embedding(image)

    # Save image locally
    if image_path is None:
        image_path = file.filename
    save_path = os.path.join(SHOE_IMAGES_FOLDER, image_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    # Add embedding and path
    index.add(emb.reshape(1, -1))
    image_paths.append(save_path)

    # Persist index and paths
    save_index(index, FAISS_INDEX_PATH)
    # save image metadata in emebdding meta index.json

    return {"message": "Image added successfully", "image_path": save_path}
