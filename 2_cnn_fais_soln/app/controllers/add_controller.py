import os
import json
import string
import secrets
from fastapi import UploadFile, HTTPException
from PIL import Image
from bson import ObjectId
from app.model import extract_embedding
from app.db.mongo import products_col, embedding_cnn_faiss_metadata_col
from app.search import save_index
from app.config import SHOE_IMAGES_FOLDER,  FAISS_INDEX_PATH

class AddController:
    def __init__(
        self,
        faiss_cnn_index
    ):
        self.index = faiss_cnn_index
        self.extract_embedding = extract_embedding
        self.save_index = save_index
        self.images_folder = SHOE_IMAGES_FOLDER  # e.g. "../data/shoe_images"
        self.products_col = products_col
        self.embedding_cnn_faiss_metadata_col = embedding_cnn_faiss_metadata_col
        self.faiss_index_path = FAISS_INDEX_PATH
        self.embedding_metadata = []  # Initialize or load from file if needed

    def _generate_image_id(self, length=7):
        alphabet = string.ascii_uppercase + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    async def add_product(
        self,
        item_id: str,
        product_type: list,
        item_name: list,
        main_image: UploadFile,
        other_images: list = None,
    ):
        if not main_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Main image must be an image")

        # --- Main image ---
        main_image_id = self._generate_image_id()
        main_image_path = await self._save_image(main_image, main_image_id)
        main_image_emb = self._extract_embedding_from_path(main_image_path)
        main_image_rel_path = self._get_relative_image_path(main_image_path)

        # Get FAISS index for main image (before adding)
        main_faiss_index = self.index.ntotal
        self.index.add(main_image_emb.reshape(1, -1))

        main_image_meta = {
            "faiss_index": main_faiss_index,
            "image_id": main_image_id,
            "image_path": main_image_rel_path,
            "item_id": item_id,
        }

        # --- Other images ---
        other_image_metas = []
        other_image_embs = []
        if other_images:
            for img_file in other_images:
                if not img_file.content_type.startswith("image/"):
                    raise HTTPException(status_code=400, detail="One of the other images is not an image")
                img_id = self._generate_image_id()
                img_path = await self._save_image(img_file, img_id)
                img_emb = self._extract_embedding_from_path(img_path)
                img_rel_path = self._get_relative_image_path(img_path)

                # Get FAISS index for this image before adding
                img_faiss_index = self.index.ntotal
                self.index.add(img_emb.reshape(1, -1))

                other_image_metas.append({
                    "faiss_index": img_faiss_index,
                    "image_id": img_id,
                    "image_path": img_rel_path,
                    "item_id": item_id,
                })
                other_image_embs.append(img_emb)

        # Update embedding metadata in-memory and MongoDB
        self.embedding_metadata.append(main_image_meta)
        self.embedding_metadata.extend(other_image_metas)

        self.save_index(self.index, self.faiss_index_path)

        self.embedding_cnn_faiss_metadata_col.insert_many([main_image_meta] + other_image_metas)

        # Insert product metadata into MongoDB
        product_doc = {
            "item_id": item_id,
            "product_type": product_type,
            "item_name": item_name,
            "main_image_id": main_image_id,
            "other_image_id": [m["image_id"] for m in other_image_metas],
        }
        self.products_col.insert_one(product_doc)

        return {
            "message": "Product added successfully",
            "item_id": item_id,
            "main_image_id": main_image_id,
            "other_image_ids": [m["image_id"] for m in other_image_metas],
        }

    async def _save_image(self, file: UploadFile, image_id: str) -> str:
        ext = os.path.splitext(file.filename)[1]
        filename = f"{image_id}{ext}"

        save_dir = os.path.join(self.images_folder, "new")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)

        return save_path  # Return absolute path

    def _get_relative_image_path(self, absolute_path: str) -> str:
        # Return path relative to self.images_folder (e.g. "new/XXXXX.jpg")
        return os.path.relpath(absolute_path, self.images_folder).replace("\\", "/")

    def _extract_embedding_from_path(self, image_path: str):
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            emb = self.extract_embedding(image)
            return emb
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image {image_path}: {str(e)}")

    def _convert_objectid_to_str(self, obj):
        if isinstance(obj, list):
            return [self._convert_objectid_to_str(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self._convert_objectid_to_str(v) for k, v in obj.items()}
        if isinstance(obj, ObjectId):
            return str(obj)
        return obj
