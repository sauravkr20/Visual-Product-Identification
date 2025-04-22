import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from sklearn.cluster import MiniBatchKMeans
import pickle
from app.db.mongo import embedding_cnn_faiss_metadata_col, products_col
from app.model import extract_embedding
from app.search import build_faiss_index, save_index
from app.config import (
    FAISS_INDEX_PATH,
    FAISS_HYBRID_INDEX_PATH,
    IMAGE_PATHS_JSON,
    SHOE_IMAGES_FOLDER,
    KMEANS_MODEL_PATH,
    SHOE_PRODUCT_JSON_PATH
)

def build_products_col():
    print("Loading product data from JSON...")
    with open(SHOE_PRODUCT_JSON_PATH, "r") as f:
        products = json.load(f)

    if not products:
        print("No products found in JSON file.")
        return

    # Deduplicate products by 'item_id'
    unique_products = {}
    for product in products:
        item_id = product.get("item_id")
        if item_id and item_id not in unique_products:
            unique_products[item_id] = product

    deduped_products = list(unique_products.values())
    print(f"Original products count: {len(products)}")
    print(f"Deduplicated products count: {len(deduped_products)}")

    print(f"Inserting {len(deduped_products)} products into MongoDB...")
    
    products_col.delete_many({})
    products_col.insert_many(deduped_products)

    # Create an index on 'item_id' for fast lookups
    products_col.create_index("item_id", unique=True)
    print("Product data inserted successfully with index on 'item_id'.")


def build_cnn_faiss_index():
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    embeddings = []
    metadata_docs = []
    print(f"Processing {len(original_metadata)} images for CNN FAISS index...")

    for idx, record in enumerate(original_metadata):
        relative_path = Path(record["image_path"])
        image_id = record["image_id"]
        item_id = record["item_id"]
        image_file_path = Path(SHOE_IMAGES_FOLDER) / relative_path

        if not image_file_path.exists():
            print(f"Image not found: {image_file_path}")
            continue

        try:
            image = Image.open(image_file_path).convert("RGB")
            emb = extract_embedding(image)
            embeddings.append(emb)

            metadata_docs.append({
                "faiss_index": idx,
                "image_id": image_id,
                "item_id": item_id,
                "image_path": str(relative_path)
            })

        except Exception as e:
            print(f"Failed to process {relative_path}: {e}")

        if (idx + 1) % 100 == 0 or (idx + 1) == len(original_metadata):
            print(f"Processed {idx + 1}/{len(original_metadata)} images")

    if not embeddings:
        print("No embeddings extracted. Exiting CNN FAISS build.")
        return

    # Clear existing metadata and insert all at once
    embedding_cnn_faiss_metadata_col.delete_many({})
    if metadata_docs:
        embedding_cnn_faiss_metadata_col.insert_many(metadata_docs)
        embedding_cnn_faiss_metadata_col.create_index("faiss_index")

    embeddings = np.stack(embeddings).astype("float32")
    index = build_faiss_index(embeddings)
    save_index(index, FAISS_INDEX_PATH)

    print(f"CNN FAISS index saved to {FAISS_INDEX_PATH}")


# def extract_sift_descriptors(image):
#     gray = np.array(image.convert("L"))
#     sift = cv2.SIFT_create()
#     _, descriptors = sift.detectAndCompute(gray, None)
#     return descriptors if descriptors is not None else np.zeros((1, 128), dtype=np.float32)


# def build_cnn_sift_hybrid_index(k=64, batch_size=1000):
#     with open(IMAGE_PATHS_JSON, "r") as f:
#         original_metadata = json.load(f)

#     cnn_embeddings = []
#     sift_descriptors_all = []
#     metadata_docs = []

#     print(f"Extracting CNN and SIFT features for {len(original_metadata)} images...")

#     for idx, record in enumerate(original_metadata):
#         relative_path = Path(record["image_path"])
#         image_id = record["image_id"]
#         item_id = record["item_id"]
#         image_file_path = Path(SHOE_IMAGES_FOLDER) / relative_path

#         if not image_file_path.exists():
#             print(f"Image not found: {image_file_path}")
#             continue

#         try:
#             image = Image.open(image_file_path).convert("RGB")
#             cnn_emb = extract_embedding(image)
#             cnn_embeddings.append(cnn_emb)

#             sift_desc = extract_sift_descriptors(image)
#             sift_descriptors_all.append(sift_desc)

#             metadata_docs.append({
#                 "faiss_index": idx,
#                 "image_id": image_id,
#                 "item_id": item_id,
#                 "image_path": str(relative_path)
#             })

#         except Exception as e:
#             print(f"Failed to process {relative_path}: {e}")

#         if (idx + 1) % 100 == 0 or (idx + 1) == len(original_metadata):
#             print(f"Extracted features for {idx + 1}/{len(original_metadata)} images")

#     if not cnn_embeddings or not sift_descriptors_all:
#         print("No features extracted. Exiting hybrid index build.")
#         return

#     all_sift_desc = np.vstack(sift_descriptors_all).astype(np.float32)

#     print(f"Fitting KMeans with k={k} on {all_sift_desc.shape[0]} SIFT descriptors...")
#     kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1)
#     kmeans.fit(all_sift_desc)

#     with open(KMEANS_MODEL_PATH, "wb") as f:
#         pickle.dump(kmeans, f)
#     print(f"KMeans model saved to {KMEANS_MODEL_PATH}")

#     sift_histograms = []
#     for sift_desc in sift_descriptors_all:
#         words = kmeans.predict(sift_desc)
#         hist = np.bincount(words, minlength=k).astype(np.float32)
#         hist /= (hist.sum() + 1e-7)
#         sift_histograms.append(hist)

#     cnn_embeddings = np.stack(cnn_embeddings)
#     sift_histograms = np.stack(sift_histograms)
#     hybrid_embeddings = np.hstack([cnn_embeddings, sift_histograms])

#     embedding_meta_hybrid_col.delete_many({})
#     if metadata_docs:
#         embedding_meta_hybrid_col.insert_many(metadata_docs)
#         embedding_meta_hybrid_col.create_index("faiss_index")

#     print(f"Building FAISS index for hybrid embeddings with shape {hybrid_embeddings.shape}...")
#     index = build_faiss_index(hybrid_embeddings)
#     save_index(index, FAISS_HYBRID_INDEX_PATH)

#     print(f"Hybrid FAISS index saved to {FAISS_HYBRID_INDEX_PATH}")


