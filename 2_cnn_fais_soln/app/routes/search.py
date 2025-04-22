from fastapi import APIRouter, UploadFile, File, Depends
from app.models.search_models import SearchRequest, SearchResponse
from app.controllers.search_controller import SearchController

router = APIRouter()

search_controller: SearchController = None  # Initialized in main.py

@router.post("/search/", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    params: SearchRequest = Depends()
):
    results = await search_controller.search(file, params)
    return SearchResponse(results=results)
