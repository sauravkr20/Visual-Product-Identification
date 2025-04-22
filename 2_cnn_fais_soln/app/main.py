from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import SHOE_IMAGES_FOLDER, FAISS_INDEX_PATH, SHOE_PRODUCT_JSON_PATH, EMBEDDING_META_INDEX
from app.model import extract_embedding
from app.search import load_index, load_embedding_metadata, save_index, search, load_product_metadata

from app.data_loading import load_and_transform_data

from app.services.cnn_faiss import CNNFaissSearch
from app.controllers.search_controller import SearchController
from app.controllers.products_controller import ProductsController
from app.controllers.add_controller import AddController

from app.routes import search as search_routes
from app.routes import products as products_routes
from app.routes import add as add_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and embedding metadata
index = load_index(FAISS_INDEX_PATH)
embedding_metadata = load_embedding_metadata(EMBEDDING_META_INDEX)

# Load and transform product data
transformed_products, product_dict = load_and_transform_data(
    SHOE_PRODUCT_JSON_PATH,
    EMBEDDING_META_INDEX,
)

# Mount static files for images
app.mount("/images", StaticFiles(directory=SHOE_IMAGES_FOLDER), name="images")

# Initialize services and controllers
cnn_faiss_service = CNNFaissSearch(index, embedding_metadata, extract_embedding, search)
search_controller = SearchController(cnn_faiss_service)
products_controller = ProductsController(product_dict)
add_controller = AddController(index, embedding_metadata, extract_embedding, save_index, SHOE_IMAGES_FOLDER, EMBEDDING_META_INDEX)

# Inject controllers into routers
search_routes.search_controller = search_controller
products_routes.products_controller = products_controller
add_routes.add_controller = add_controller

# Include routers
app.include_router(search_routes.router)
app.include_router(products_routes.router)
app.include_router(add_routes.router)
