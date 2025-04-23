from pydantic import BaseModel
from typing import List, Dict

class ItemName(BaseModel):
    language_tag: str
    value: str

class AddProductRequest(BaseModel):
    item_id: str
    product_type: List[str]
    item_name: List[ItemName]

