from PIL import Image
from typing import List
from app.models.search_models import SearchResultItem
from app.db.mongo import embedding_cnn_faiss_metadata_col

class CNNFaissSearch:
    def __init__(self, index,  extract_embedding_func, search_func):
        self.index = index
        self.extract_embedding = extract_embedding_func
        self.search = search_func

    async def search_image(self, image: Image.Image, top_k: int) -> List[SearchResultItem]:
        emb = self.extract_embedding(image)
        indices, scores = self.search(self.index, emb, top_k)
        results = []
        for idx, score in zip(indices, scores):
            # Query embedding metadata by index or image_id
            embedding_doc = embedding_cnn_faiss_metadata_col.find_one({"faiss_index": idx})
            if not embedding_doc:
                continue
            image_id = embedding_doc["image_id"]
            image_path = embedding_doc["image_path"]
            item_id = embedding_doc.get("item_id")

            results.append({
                "image_id": image_id,
                "item_id": item_id,
                "image_path": image_path,
                "score": float(score)
            })

        return results
