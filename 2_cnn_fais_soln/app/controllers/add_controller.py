from fastapi import UploadFile, HTTPException
from PIL import Image
import io
import os
import json

class AddController:
    def __init__(self, index, embedding_metadata, extract_embedding_func, save_index_func, images_folder, embedding_meta_path):
        self.index = index
        self.embedding_metadata = embedding_metadata
        self.extract_embedding = extract_embedding_func
        self.save_index = save_index_func
        self.images_folder = images_folder
        self.embedding_meta_path = embedding_meta_path

    async def add_image(self, file: UploadFile, image_path: str = None):
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        emb = self.extract_embedding(image)

        if image_path is None:
            image_path = file.filename
        save_path = os.path.join(self.images_folder, image_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        # Add embedding and metadata
        self.index.add(emb.reshape(1, -1))
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        new_meta = {"image_id": image_id, "image_path": image_path}
        self.embedding_metadata.append(new_meta)

        # Persist index and metadata
        self.save_index(self.index, FAISS_INDEX_PATH)
        with open(self.embedding_meta_path, "w") as f:
            json.dump(self.embedding_metadata, f, indent=2)

        return {"message": "Image added successfully", "image_path": save_path, "image_id": image_id}
