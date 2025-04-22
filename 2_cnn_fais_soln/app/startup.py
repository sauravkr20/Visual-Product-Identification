import json
import numpy as np
from pathlib import Path
from PIL import Image
from app.model import extract_embedding
from app.search import build_faiss_index, save_index
from app.config import FAISS_INDEX_PATH, IMAGE_PATHS_JSON, EMBEDDING_META_INDEX, SHOE_IMAGES_FOLDER

def build_index_from_folder():
    # Load original image metadata
    with open(IMAGE_PATHS_JSON, "r") as f:
        original_metadata = json.load(f)

    embeddings = []
    embedding_meta = []
    print(f"Processing {len(original_metadata)} images...")

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
        print("No embeddings extracted. Exiting.")
        return

    embeddings = np.stack(embeddings).astype("float32")
    index = build_faiss_index(embeddings)
    save_index(index, FAISS_INDEX_PATH)

    # Save the metadata aligned with embeddings
    with open(EMBEDDING_META_INDEX, "w") as f:
        json.dump(embedding_meta, f, indent=2)

    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Embedding metadata saved to {EMBEDDING_META_INDEX}")
