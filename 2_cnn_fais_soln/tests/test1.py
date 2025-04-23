import os
import json
from pymongo import MongoClient
import time
import asyncio
import shutil

from starlette.datastructures import UploadFile
from app.config import SHOE_IMAGES_FOLDER, TEST_SET_MODIFY_FOLDER, FAISS_INDEX_PATH
from app.models.search_models import SearchRequest
from app.controllers.search_controller import SearchController
from app.search import load_index, search
from app.model import extract_embedding
from app.services.cnn_faiss import CNNFaissSearch
from tests.test_modification import apply_modification

import numpy as np



LOG_FILE_PATH = "search_accuracy_test.log"

# Setup MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["visual_product_db"]
products_col = db["products"]
embedding_cnn_faiss_metadata_col = db["embedding_cnn_faiss_metadata"]

# Setup search controller
index = load_index(FAISS_INDEX_PATH)
cnn_faiss_service = CNNFaissSearch(index, extract_embedding, search)  # Pass None or real search fn if needed
search_controller = SearchController(cnn_faiss_service)

os.makedirs(TEST_SET_MODIFY_FOLDER, exist_ok=True)

async def create_upload_file_from_path(file_path: str) -> UploadFile:
    file = open(file_path, "rb")
    file_size = os.path.getsize(file_path)
    content_type = "image/jpeg"  # Assume JPEG, adjust if needed
    headers = {"content-type": content_type}
    return UploadFile(
        filename=os.path.basename(file_path),
        file=file,
        size=file_size,
        headers=headers
    )

def apply_nothing(input_image_path, output_image_path):
    if not os.path.exists(output_image_path):
        shutil.copy2(input_image_path, output_image_path)


async def run_search_test():
    # Sample 100 products
    sampled_products = list(products_col.aggregate([{"$sample": {"size": 100}}, {"$project": {"item_id": 1, "main_image_id": 1}}]))
    sample_main_image_ids = [p["main_image_id"] for p in sampled_products]

    # Fetch main images metadata for these products (get one image per ID)
    main_images = []
    for image_id in sample_main_image_ids:
        image_doc = embedding_cnn_faiss_metadata_col.find_one({"image_id": image_id})
        if image_doc:
            main_images.append(image_doc)

    total = len(main_images)
    print(f"Sampled {total} products for testing.")

    pass_count_original = 0
    pass_count_modified = 0
    log_entries = []

    total_search_duration = 0
    start_time = time.time()

    for idx, img_meta in enumerate(main_images, 1):
        item_id = img_meta["item_id"]
        image_rel_path = img_meta["image_path"]
        image_abs_path = os.path.join(SHOE_IMAGES_FOLDER, image_rel_path)

        if not os.path.exists(image_abs_path):
            print(f"[{idx}/{total}] Image file not found: {image_abs_path}")
            continue

        # Prepare UploadFile for original image
        upload_file_original = await create_upload_file_from_path(image_abs_path)
        params = SearchRequest()  # Customize if needed

        # --- Original image search ---
        search_start_time = time.time()
        results = await search_controller.search(upload_file_original, params)
        upload_file_original.file.close()  # Close the file after use
        original_search_duration = time.time() - search_start_time
        total_search_duration += original_search_duration

        original_matched_item_ids = [res.get("item_id") for res in results]


        found_original = any(res.get("item_id") == item_id for res in results)
        if found_original:
            pass_count_original += 1

        # Prepare modified image
        modified_image_path = os.path.join(TEST_SET_MODIFY_FOLDER, f"{item_id}_copy.jpg")
        apply_modification(image_abs_path, modified_image_path)

        # Prepare UploadFile for modified image
        upload_file_modified = await create_upload_file_from_path(modified_image_path)

        # --- Modified image search ---
        search_start_time = time.time()
        modified_results = await search_controller.search(upload_file_modified, params)
        upload_file_modified.file.close()
        modified_search_duration = time.time() - search_start_time
        total_search_duration += modified_search_duration

        modified_matched_item_ids = [res.get("item_id") for res in modified_results]

        found_modified = any(res.get("item_id") == item_id for res in modified_results)
        if found_modified:
            pass_count_modified += 1

        log_entries.append({
            "item_id": item_id,
            "found_original": found_original,
            "found_modified": found_modified,
            "original_image_path": image_abs_path,
            "modified_image_path": modified_image_path,
            "original_search_duration": original_search_duration , 
            "modified_search_duration": modified_search_duration, 
            "original_matched_item_ids": original_matched_item_ids,
            "modified_matched_item_ids": modified_matched_item_ids
        })

        print(f"[{idx}/{total}] Item {item_id} - Original found: {found_original}, Modified found: {found_modified} (Original Search time: {original_search_duration:.4f}s)  (modified Search time: {modified_search_duration:.4f}s)")

    duration = time.time() - start_time
    original_rate = (pass_count_original / total) * 100 if total else 0
    modified_rate = (pass_count_modified / total) * 100 if total else 0
    avg_search_duration = total_search_duration / (total * 2) if total else 0  # divide by total no. of searches = total *2

    print(f"\nOriginal images pass rate: {pass_count_original}/{total} = {original_rate:.2f}%")
    print(f"Modified images pass rate: {pass_count_modified}/{total} = {modified_rate:.2f}%")
    print(f"Test duration: {duration:.2f} seconds (Average search duration: {avg_search_duration:.4f}s)")

    # Write detailed log
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write(f"Search Accuracy Test Log\n")
        log_file.write(f"Total products tested: {total}\n")
        log_file.write(f"Original images pass rate: {original_rate:.2f}%\n")
        log_file.write(f"Modified images pass rate: {modified_rate:.2f}%\n")
        log_file.write(f"Total test duration: {duration:.2f} seconds (Average search duration: {avg_search_duration:.4f}s)\n")
        log_file.write("Details per product:\n")
        for entry in log_entries:
            log_file.write(json.dumps(entry) + "\n")

    print(f"Detailed log saved to {LOG_FILE_PATH}")

def main():
    asyncio.run(run_search_test())

if __name__ == "__main__":
    main()
