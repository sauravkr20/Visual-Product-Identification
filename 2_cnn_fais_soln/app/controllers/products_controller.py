from fastapi import HTTPException
from typing import Optional, List, Dict
from app.db.mongo import products_col, embedding_cnn_faiss_metadata_col

class ProductsController:
    def __init__(self):
        pass

    async def get_product(self, item_id: str) -> Optional[dict]:
        product = self._fetch_product(item_id)
        image_ids = self._collect_image_ids(product)
        embedding_dict = self._fetch_embedding_metadata(image_ids)
        transformed_product = self._transform_product(product, embedding_dict)
        return transformed_product

    def _fetch_product(self, item_id: str) -> dict:
        print("Fetching product with item_id:", item_id)
        product = products_col.find_one({"item_id": item_id})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        product.pop("_id", None)  # Remove MongoDB internal _id field
        return product

    def _collect_image_ids(self, product: dict) -> List[str]:
        image_ids = []
        main_img_id = product.get("main_image_id")
        if main_img_id:
            image_ids.append(main_img_id)
        other_img_ids = product.get("other_image_id", [])
        image_ids.extend(other_img_ids)
        return image_ids

    def _fetch_embedding_metadata(self, image_ids: List[str]) -> Dict[str, dict]:
        embedding_docs = embedding_cnn_faiss_metadata_col.find({"image_id": {"$in": image_ids}})
        return {doc["image_id"]: doc for doc in embedding_docs}

    def _transform_product(self, product: dict, embedding_dict: Dict[str, dict]) -> dict:
        main_img_id = product.get("main_image_id")
        main_image = {
            "image_id": main_img_id,
            "image_path": embedding_dict.get(main_img_id, {}).get("image_path", "") if main_img_id else ""
        }

        other_images = []
        for img_id in product.get("other_image_id", []):
            img_path = embedding_dict.get(img_id, {}).get("image_path", "")
            if img_path:
                other_images.append({"image_id": img_id, "image_path": img_path})

        return {
            "item_id": product["item_id"],
            "product_type": product.get("product_type", []),
            "item_name": product.get("item_name", []),
            "main_image": main_image,
            "other_images": other_images,
        }
