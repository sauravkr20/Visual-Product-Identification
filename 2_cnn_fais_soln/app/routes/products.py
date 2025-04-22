from fastapi import APIRouter
from app.controllers.products_controller import ProductsController

router = APIRouter()

products_controller: ProductsController = None  # Initialized in main.py

@router.get("/products/{item_id}")
async def get_product(item_id: str):
    return await products_controller.get_product(item_id)
