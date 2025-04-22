from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["visual_product_db"]

products_col = db["products"]
embedding_cnn_faiss_metadata_col = db["embedding_cnn_faiss_metadata"]
