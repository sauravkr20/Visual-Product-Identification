import os
from dotenv import load_dotenv

load_dotenv()  # load from .env

SHOE_IMAGES_FOLDER = os.getenv("SHOE_IMAGES_FOLDER")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
EMBEDDING_META_INDEX=os.getenv("EMBEDDING_META_INDEX")
SHOE_PRODUCT_JSON_PATH=os.getenv("SHOE_PRODUCT_JSON_PATH")
