from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import SHOE_IMAGES_FOLDER, FAISS_INDEX_PATH, SHOE_PRODUCT_JSON_PATH, EMBEDDING_META_INDEX
from app.model import extract_embedding
from app.search import load_index, load_embedding_metadata, save_index, search, load_product_metadata

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = load_index(FAISS_INDEX_PATH)
products = load_product_metadata(SHOE_PRODUCT_JSON_PATH)
embedded_index_list = load_embedding_metadata(EMBEDDING_META_INDEX)  # List of dicts with image_id, image_path

# Convert embedded_index_list to dict for fast lookup: image_id -> image_path
embedded_index_dict = {item["image_id"]: item["image_path"] for item in embedded_index_list}

# Transform product data once, resolving image paths
transformed_products = []
for product in products:
    main_img_id = product.get("main_image_id")
    main_image_path = embedded_index_dict.get(main_img_id, "")
    
    other_images = []
    for img_id in product.get("other_image_id", []):
        path = embedded_index_dict.get(img_id, "")
        if path:
            other_images.append({"image_id": img_id, "image_path": path})

    transformed_products.append({
        "item_id": product["item_id"],
        "product_type": product.get("product_type", []),
        "item_name": product.get("item_name", []),
        "main_image": {
            "image_id": main_img_id,
            "image_path": main_image_path,
        },
        "other_images": other_images,
    })

product_dict = {p["item_id"]: p for p in transformed_products}

app.mount("/images", StaticFiles(directory=SHOE_IMAGES_FOLDER), name="images")

@app.post("/search/")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    emb = extract_embedding(image)
    indices, scores = search(index, emb, top_k)
    results = [{"image_path": embedded_index_list[i], "score": scores[idx]} for idx, i in enumerate(indices)]
    return {"results": results}

@app.get("/products/{item_id}")
async def get_product(item_id: str):
    product = next((p for p in transformed_products if p['item_id'] == item_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

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
    embedded_index_list.append(save_path)

    # Persist index and paths
    save_index(index, FAISS_INDEX_PATH)
    # save image metadata in emebdding meta index.json

    return {"message": "Image added successfully", "image_path": save_path}
