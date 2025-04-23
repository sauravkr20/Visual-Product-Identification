to run 

    uvicorn app.main:app --reload --port 5000



## Why CNN + FAISS?

We use **CNN** to make vector embedding from shoe images. CNN help get important features from image, not just pixels.  
Then use **FAISS** to store these vectors and search fast. FAISS good for big data, fast search, can handle millions vectors.  
Together, CNN + FAISS make system that find similar shoes fast and accurate.

## Backend Routes and What They Do

## 1. Add Product Route
- Receive product info + images.
- Save images on disk.
- Extract embedding from saved images using CNN.
- Add embedding to FAISS index with correct `faiss_index`.
- Save metadata in MongoDB with relative image path and faiss_index.

Example snippet from controller:

python
```
main_faiss_index = self.index.ntotal
self.index.add(main_image_emb.reshape(1, -1))
main_image_meta = {
    "faiss_index": main_faiss_index,
    "image_id": main_image_id,
    "image_path": relative_path,
    "item_id": item_id,
}
self.embedding_cnn_faiss_metadata_col.insert_one(main_image_meta)

```
## 2. Search Route
- Async FastAPI endpoint.
- Accept image upload.
- Extract embedding from uploaded image.
- Search FAISS index for nearest neighbors.
- Return matching products with `item_id`.
   
Example usage:

python
```
results = await search_controller.search(upload_file, params)
```


## 3. Build Index Script

- Process images in batches (e.g. 1000).
- Extract embeddings batch-wise.
- Insert metadata batch-wise to MongoDB.
- Build or update FAISS index with all embeddings.
- Save index to disk.

## Test Script

- Pick 100 random products from MongoDB.
- Get main image metadata from embedding collection.
- For each image:
    - Run search on original image.
    - Copy image (no change) to test folder.
    - Run search on copied image.
    - Check if search results contain same `item_id`.
- Calculate pass rate for original and copied images.
- Save detailed log file.

Example snippet for testing search:

python
```
upload_file = await create_upload_file_from_path(image_abs_path)
results = await search_controller.search(upload_file, params)
found = any(r["item_id"] == item_id for r in results)

```

Log saved like:

text
```
Item ID: B079V42VTT, Original found: True, Modified found: True
...
Original images pass rate: 95%
Modified images pass rate: 93%

```

## Summary

- CNN make embedding from images.
- FAISS store embeddings and search fast.
- MongoDB save product and image metadata.
- Add product save images, make embedding, add to FAISS and Mongo.
- Search route async accept image upload, search FAISS, return matches.
- Build index batch process images and embeddings.
- Test script check search accuracy on original and copied images with logs.

If want help with install or usage, just ask!