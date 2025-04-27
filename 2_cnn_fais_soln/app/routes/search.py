from fastapi import APIRouter, Form, UploadFile, File, Depends
from app.models.search_models import SearchRequest, SearchResponse
from app.controllers.search_controller import SearchController

router = APIRouter()

search_controller: SearchController = None  # Initialized in main.py
@router.post("/search/", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    method: str = Form("cnn_faiss", description="Search method: cnn_faiss or clip_faiss"),
    top_k: int = Form(5, ge=1, le=50, description="Number of top results to return")
):
    print(f"Received search request: method={method}, top_k={top_k}")
    class Params:
        def __init__(self, method, top_k):
            self.method = method
            self.top_k = top_k

    params = Params(method, top_k)
    results = await search_controller.search(file, params)
    return SearchResponse(results=results)