from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import json

from app.controllers.add_controller import AddController

router = APIRouter()

add_controller:AddController = None 


@router.post("/add_product")
async def add_product(
    item_id: str = Form(...),
    product_type: str = Form(...),
    item_name: str = Form(...),  # can be JSON string or plain string
    main_image: UploadFile = File(...),
    other_images: Optional[List[UploadFile]] = File(None),
):
    # Try to parse item_name as JSON list
    try:
        parsed = json.loads(item_name)
        if isinstance(parsed, list):
            item_name_list = parsed
        else:
            # If JSON parsed but not a list, wrap in list
            item_name_list = [parsed]
    except json.JSONDecodeError:
        # Not JSON, treat as plain string, wrap in list with default language_tag
        item_name_list = [{"language_tag": "en", "value": item_name}]

    # Wrap product_type string in list for your add_product method
    product_type_list = [product_type]

    result = await add_controller.add_product(
        item_id=item_id,
        product_type=product_type_list,
        item_name=item_name_list,
        main_image=main_image,
        other_images=other_images,
    )
    return result

