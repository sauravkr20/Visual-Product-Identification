import os
import json
import time
import asyncio
from pymongo import MongoClient
from starlette.datastructures import UploadFile
from app.models.search_models import SearchRequest
from app.controllers.search_controller import SearchController
from app.model import extract_embedding, extract_clip_embedding
from app.services.cnn_faiss import CNNFaissSearch
from app.services.clip_faiss import CLIPFaissSearch
from app.search import load_index, search
from app.config import SHOE_IMAGES_FOLDER, TEST_SET_MODIFY_FOLDER, FAISS_INDEX_PATH, CLIP_FAISS_INDEX_PATH
from tests.test_modification import apply_modification

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["visual_product_db"]
products_col = db["products"]
embedding_cnn_faiss_metadata_col = db["embedding_cnn_faiss_metadata"]
embedding_clip_faiss_metadata_col = db["embedding_clip_faiss_metadata"]

# Load FAISS indexes
cnn_index = load_index(FAISS_INDEX_PATH)
clip_index = load_index(CLIP_FAISS_INDEX_PATH)

# Initialize services
cnn_faiss_service = CNNFaissSearch(cnn_index, extract_embedding, search)  # Replace None with actual search fn if needed
clip_faiss_service = CLIPFaissSearch(clip_index, extract_clip_embedding, search)  # Replace None with actual search fn if needed

# Initialize controller with both services
search_controller = SearchController(cnn_faiss_service, clip_faiss_service)

# Ask for test case name
test_case_name = input("Enter test case name (folder name) to save modified images and logs: ").strip()
modified_images_folder = os.path.join(TEST_SET_MODIFY_FOLDER, test_case_name)
os.makedirs(modified_images_folder, exist_ok=True)
print(f"Modified images and logs will be saved to: {modified_images_folder}")

# Helper to create UploadFile from path
async def create_upload_file_from_path(file_path: str) -> UploadFile:
    file = open(file_path, "rb")
    file_size = os.path.getsize(file_path)
    headers = {"content-type": "image/jpeg"}
    return UploadFile(
        filename=os.path.basename(file_path),
        file=file,
        size=file_size,
        headers=headers
    )

# Sample images once
def sample_images(sample_size=100):
    sampled_products = list(products_col.aggregate([
        {"$sample": {"size": sample_size}},
        {"$project": {"item_id": 1, "main_image_id": 1}}
    ]))
    sample_main_image_ids = [p["main_image_id"] for p in sampled_products]

    main_images = []
    for image_id in sample_main_image_ids:
        doc = embedding_cnn_faiss_metadata_col.find_one({"image_id": image_id})
        if doc:
            main_images.append(doc)
    return main_images

# Generic test function for CNN or CLIP
async def run_search_test(search_controller, metadata_col, log_file_path, method_name):
    main_images = sample_images()
    total = len(main_images)
    print(f"[{method_name}] Sampled {total} images for testing.")

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
            print(f"[{method_name}] [{idx}/{total}] Image file not found: {image_abs_path}")
            continue

        upload_file_original = await create_upload_file_from_path(image_abs_path)
        # Create SearchRequest params instance
        params = SearchRequest(method=method_name, top_k=5)

        # Original image search
        search_start = time.time()
        results = await search_controller.search(upload_file_original, params)
        upload_file_original.file.close()
        duration_original = time.time() - search_start
        total_search_duration += duration_original

        print("results",results)
        found_original = any(r.get('item_id') == item_id for r in results)
        if found_original:
            pass_count_original += 1

        # Modified image (copy)
        modified_image_path = os.path.join(modified_images_folder, f"{item_id}_copy.jpg")
        apply_modification(image_abs_path, modified_image_path)
        upload_file_modified = await create_upload_file_from_path(modified_image_path)

        # Modified image search
        search_start = time.time()
        modified_results = await search_controller.search(upload_file_modified, params)
        upload_file_modified.file.close()
        duration_modified = time.time() - search_start
        total_search_duration += duration_modified

        found_modified = any(r.get('item_id') == item_id for r in modified_results)
        if found_modified:
            pass_count_modified += 1

        log_entries.append({
            "item_id": item_id,
            "found_original": found_original,
            "found_modified": found_modified,
            "original_image_path": image_abs_path,
            "modified_image_path": modified_image_path,
            "search_duration_original": duration_original,
            "search_duration_modified": duration_modified,
            "original_matched_item_ids": [r.get('item_id') for r in results],
            "modified_matched_item_ids": [r.get('item_id') for r in modified_results],
        })

        print(f"[{method_name}] [{idx}/{total}] Item {item_id} - Original found: {found_original}, Modified found: {found_modified} (Durations: {duration_original:.3f}s, {duration_modified:.3f}s)")

    total_duration = time.time() - start_time
    total_searches = total * 2  # original + modified per image
    avg_search_duration = total_search_duration / total_searches if total_searches else 0

    original_rate = (pass_count_original / total) * 100 if total else 0
    modified_rate = (pass_count_modified / total) * 100 if total else 0

    print(f"\n[{method_name}] Original images pass rate: {pass_count_original}/{total} = {original_rate:.2f}%")
    print(f"[{method_name}] Modified images pass rate: {pass_count_modified}/{total} = {modified_rate:.2f}%")
    print(f"[{method_name}] Test duration: {total_duration:.2f}s (Average search duration: {avg_search_duration:.4f}s)")

    with open(log_file_path, "w") as log_file:
        log_file.write(f"{method_name} Search Accuracy Test Log\n")
        log_file.write(f"Total images tested: {total}\n")
        log_file.write(f"Original images pass rate: {original_rate:.2f}%\n")
        log_file.write(f"Modified images pass rate: {modified_rate:.2f}%\n")
        log_file.write(f"Total test duration: {total_duration:.2f}s (Average search duration: {avg_search_duration:.4f}s)\n")
        log_file.write("Details per image:\n")
        for entry in log_entries:
            log_file.write(json.dumps(entry) + "\n")

    print(f"[{method_name}] Detailed log saved to {log_file_path}")

async def main():
    # Run CNN test
    await run_search_test(
        search_controller=search_controller,
        metadata_col=embedding_cnn_faiss_metadata_col,
        log_file_path=os.path.join(modified_images_folder, f"{test_case_name}_cnn.log"),
        method_name="cnn_faiss"
    )

    # Run CLIP test
    await run_search_test(
        search_controller=search_controller,
        metadata_col=embedding_clip_faiss_metadata_col,
        log_file_path=os.path.join(modified_images_folder, f"{test_case_name}_clip.log"),
        method_name="clip_faiss"
    )

if __name__ == "__main__":
    asyncio.run(main())
