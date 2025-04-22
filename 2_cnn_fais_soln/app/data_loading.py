from typing import List, Dict
from app.search import load_embedding_metadata, load_product_metadata

def load_and_transform_data(
    product_json_path: str,
    embedding_meta_index_path: str,
) -> (List[Dict]|Dict[str, Dict]):

    products = load_product_metadata(product_json_path)
    embedding_metadata = load_embedding_metadata(embedding_meta_index_path)

    embedding_metadata_dict = {item["image_id"]: item for item in embedding_metadata}

    transformed_products = []
    for product in products:
        main_img_id = product.get("main_image_id")
        main_image_path = embedding_metadata_dict.get(main_img_id, {}).get("image_path", "")

        other_images = []
        for img_id in product.get("other_image_id", []):
            path = embedding_metadata_dict.get(img_id, {}).get("image_path", "")
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

    return transformed_products, product_dict
