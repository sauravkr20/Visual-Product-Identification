import os
from dotenv import load_dotenv

load_dotenv()  # load from .env

SHOE_IMAGES_FOLDER = os.getenv("SHOE_IMAGES_FOLDER")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
IMAGE_PATHS_JSON = os.getenv("IMAGE_PATHS_JSON")
EMBEDDING_META_INDEX=os.getenv("EMBEDDING_META_INDEX")
