from pymongo import MongoClient
from app.config import EMBEDDING_CLIP_FAISS_METADATA_COLLECTION

client = MongoClient("mongodb://localhost:27017")
db = client["visual_product_db"]

products_col = db["products"]
embedding_cnn_faiss_metadata_col = db["embedding_cnn_faiss_metadata"]
embedding_clip_faiss_metadata_col = db[EMBEDDING_CLIP_FAISS_METADATA_COLLECTION]  