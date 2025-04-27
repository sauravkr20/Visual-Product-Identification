from PIL import Image
from typing import List
from app.models.search_models import SearchResultItem
from app.db.mongo import embedding_clip_faiss_metadata_col
import numpy as np

class CLIPFaissSearch:
    def __init__(self, index, extract_clip_embedding, search_func):
        self.index = index
        self.extract_embedding = extract_clip_embedding
        self.search = search_func

    async def search_image(self, image: Image.Image, top_k: int) -> List[SearchResultItem]:
        # Extract embedding (assumed synchronous)
        emb = self.extract_embedding(image)  # numpy array shape (dim,)
        emb = emb.reshape(1, -1).astype('float32')  # FAISS expects 2D array

        print("inside the search image clip method")

        # Perform FAISS search: FAISS returns (scores, indices)
        indices, scores = self.search(self.index, emb, top_k)
        print("the scores and indices", scores, indices)

        results: List[SearchResultItem] = []

        # Iterate over the first query's results (assuming single query)
        for idx, score in zip(indices, scores):
            # Skip invalid indices
            if idx == -1:
                continue

            embedding_doc = embedding_clip_faiss_metadata_col.find_one({"faiss_index": int(idx)})
            if not embedding_doc:
                continue

            image_id = embedding_doc.get("image_id")
            image_path = embedding_doc.get("image_path")
            item_id = embedding_doc.get("item_id")

            results.append({
                "image_id": image_id,
                "item_id": item_id,
                "image_path": image_path,
                "score": float(score)
            })

        return results
