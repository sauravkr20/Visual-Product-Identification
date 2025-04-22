from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from typing import List
from app.models.search_models import SearchRequest, SearchResultItem
from app.services.cnn_faiss import CNNFaissSearch

class SearchController:
    def __init__(self, cnn_faiss_search: CNNFaissSearch):
        self.cnn_faiss_search = cnn_faiss_search

    async def search(self, file: UploadFile, params: SearchRequest) -> List[SearchResultItem]:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if params.method == "cnn_faiss":
            return await self.cnn_faiss_search.search_image(image, params.top_k)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown search method: {params.method}")
