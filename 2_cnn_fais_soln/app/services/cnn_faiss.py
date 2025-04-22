from PIL import Image
from typing import List
from app.models.search_models import SearchResultItem

class CNNFaissSearch:
    def __init__(self, index, embedding_metadata, extract_embedding_func, search_func):
        self.index = index
        self.embedding_metadata = embedding_metadata
        self.extract_embedding = extract_embedding_func
        self.search = search_func

    async def search_image(self, image: Image.Image, top_k: int) -> List[SearchResultItem]:
        emb = self.extract_embedding(image)
        indices, scores = self.search(self.index, emb, top_k)
        results = []
        for idx, score in zip(indices, scores):
            meta = self.embedding_metadata[idx]
            results.append(SearchResultItem(
                image_id=meta["image_id"],
                item_id=meta.get("item_id"),
                image_path=meta["image_path"],
                score=float(score),
            ))
        return results
