from fastapi import HTTPException
from typing import Optional

class ProductsController:
    def __init__(self, product_dict):
        self.product_dict = product_dict

    async def get_product(self, item_id: str) -> Optional[dict]:
        print("hii product to be found")
        print(item_id)
        product = self.product_dict.get(item_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
