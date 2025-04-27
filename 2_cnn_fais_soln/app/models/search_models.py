from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SearchRequest(BaseModel):
    method: str = Field(description="Search method")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top results to return")

class SearchResultItem(BaseModel):
    image_id: str
    item_id: Optional[str]
    image_path: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
