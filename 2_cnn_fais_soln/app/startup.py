import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from sklearn.cluster import MiniBatchKMeans
import pickle

from app.model import extract_embedding
from app.search import build_faiss_index, save_index
from app.config import (
    FAISS_INDEX_PATH,
    FAISS_HYBRID_INDEX_PATH,
    IMAGE_PATHS_JSON,
    EMBEDDING_META_INDEX,
    EMBEDDING_META_HYBRID_INDEX,
    SHOE_IMAGES_FOLDER,
    KMEANS_MODEL_PATH,
)

def build_cnn_faiss_index():
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    embeddings = []
    embedding_meta = []
    print(f"Processing {len(original_metadata)} images for CNN FAISS index...")

    for idx, record in enumerate(original_metadata, 1):
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
            embedding_meta.append({
                "image_id": image_id,
                "item_id": item_id,
                "image_path": str(relative_path)
            })
        except Exception as e:
            print(f"Failed to process {relative_path}: {e}")

        if idx % 100 == 0 or idx == len(original_metadata):
            print(f"Processed {idx}/{len(original_metadata)} images")

    if not embeddings:
        print("No embeddings extracted. Exiting CNN FAISS build.")
        return

    embeddings = np.stack(embeddings).astype("float32")
    index = build_faiss_index(embeddings)
    save_index(index, FAISS_INDEX_PATH)

    with open(EMBEDDING_META_INDEX, "w") as f:
        json.dump(embedding_meta, f, indent=2)

    print(f"CNN FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"CNN embedding metadata saved to {EMBEDDING_META_INDEX}")


def extract_sift_descriptors(image):
    gray = np.array(image.convert("L"))
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros((1, 128), dtype=np.float32)
    return descriptors


def build_cnn_sift_hybrid_index(k=64, batch_size=1000):
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    cnn_embeddings = []
    sift_descriptors_all = []
    embedding_meta = []

    print(f"Extracting CNN and SIFT features for {len(original_metadata)} images...")

    # First pass: extract CNN embeddings and collect all SIFT descriptors for KMeans
    for idx, record in enumerate(original_metadata, 1):
        relative_path = Path(record["image_path"])
        image_id = record["image_id"]
        item_id = record["item_id"]
        image_file_path = Path(SHOE_IMAGES_FOLDER) / relative_path

        if not image_file_path.exists():
            print(f"Image not found: {image_file_path}")
            continue

        try:
            image = Image.open(image_file_path).convert("RGB")
            cnn_emb = extract_embedding(image)
            cnn_embeddings.append(cnn_emb)

            sift_desc = extract_sift_descriptors(image)
            sift_descriptors_all.append(sift_desc)

            embedding_meta.append({
                "image_id": image_id,
                "item_id": item_id,
                "image_path": str(relative_path)
            })

        except Exception as e:
            print(f"Failed to process {relative_path}: {e}")

        if idx % 100 == 0 or idx == len(original_metadata):
            print(f"Extracted features for {idx}/{len(original_metadata)} images")

    if not cnn_embeddings or not sift_descriptors_all:
        print("No features extracted. Exiting hybrid index build.")
        return

    # Flatten all SIFT descriptors for clustering
    all_sift_desc = np.vstack(sift_descriptors_all).astype(np.float32)

    print(f"Fitting KMeans with k={k} on {all_sift_desc.shape[0]} SIFT descriptors...")
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1)
    kmeans.fit(all_sift_desc)

    # Save KMeans model for runtime use
    with open(KMEANS_MODEL_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"KMeans model saved to {KMEANS_MODEL_PATH}")

    # Aggregate SIFT descriptors per image into BoVW histograms
    sift_histograms = []
    for sift_desc in sift_descriptors_all:
        words = kmeans.predict(sift_desc)
        hist, _ = np.histogram(words, bins=np.arange(k + 1))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        sift_histograms.append(hist)

    # Concatenate CNN embeddings and SIFT histograms
    cnn_embeddings = np.stack(cnn_embeddings).astype(np.float32)
    sift_histograms = np.stack(sift_histograms)
    hybrid_embeddings = np.hstack([cnn_embeddings, sift_histograms])

    print(f"Building FAISS index for hybrid embeddings with shape {hybrid_embeddings.shape}...")
    index = build_faiss_index(hybrid_embeddings)
    save_index(index, FAISS_HYBRID_INDEX_PATH)

    with open(EMBEDDING_META_HYBRID_INDEX, "w") as f:
        json.dump(embedding_meta, f, indent=2)

    print(f"Hybrid FAISS index saved to {FAISS_HYBRID_INDEX_PATH}")
    print(f"Hybrid embedding metadata saved to {EMBEDDING_META_HYBRID_INDEX}")

