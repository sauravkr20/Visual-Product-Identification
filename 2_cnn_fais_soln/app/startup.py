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
import time

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

BATCH_SIZE = 1000
LOG_FILE_PATH = "faiss_build_time.log"  # You can customize the log file path

def build_cnn_faiss_index():
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    total_images = len(original_metadata)
    print(f"Processing {total_images} images for CNN FAISS index in batches of {BATCH_SIZE}...")

    # Clear existing metadata before starting
    embedding_cnn_faiss_metadata_col.delete_many({})

    all_embeddings = []
    all_metadata_docs = []

    total_time_ms = 0
    batch_times = []

    for batch_start in range(0, total_images, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_images)
        batch_metadata = original_metadata[batch_start:batch_end]

        batch_embeddings = []
        batch_metadata_docs = []

        batch_start_time = time.perf_counter()

        for idx, record in enumerate(batch_metadata, start=batch_start):
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
                batch_embeddings.append(emb)

                batch_metadata_docs.append({
                    "faiss_index": idx,
                    "image_id": image_id,
                    "item_id": item_id,
                    "image_path": str(relative_path)
                })

            except Exception as e:
                print(f"Failed to process {relative_path}: {e}")

            if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
                print(f"Processed {idx + 1}/{total_images} images")

        batch_end_time = time.perf_counter()
        batch_duration_ms = (batch_end_time - batch_start_time) * 1000
        total_time_ms += batch_duration_ms
        batch_times.append(batch_duration_ms)

        print(f"Batch {batch_start} - {batch_end} processed in {batch_duration_ms:.2f} ms")
        print(f"Total time elapsed: {total_time_ms / 1000:.2f} seconds")

        if not batch_embeddings:
            print(f"No embeddings extracted in batch {batch_start} - {batch_end}. Skipping batch.")
            continue

        embedding_cnn_faiss_metadata_col.insert_many(batch_metadata_docs)

        all_embeddings.extend(batch_embeddings)
        all_metadata_docs.extend(batch_metadata_docs)

    if not all_embeddings:
        print("No embeddings extracted overall. Exiting CNN FAISS build.")
        return

    embeddings_np = np.stack(all_embeddings).astype("float32")

    index = build_faiss_index(embeddings_np)
    save_index(index, FAISS_INDEX_PATH)

    embedding_cnn_faiss_metadata_col.create_index("faiss_index")

    print(f"CNN FAISS index saved to {FAISS_INDEX_PATH} with {len(all_embeddings)} embeddings.")

    # Write timing info to log file
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write(f"Processed {total_images} images in {total_time_ms / 1000:.2f} seconds\n")
        log_file.write("Batch processing times (ms):\n")
        for i, t in enumerate(batch_times):
            log_file.write(f"Batch {i + 1}: {t:.2f} ms\n")

    print(f"Timing log saved to {LOG_FILE_PATH}")

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


