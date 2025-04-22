from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet50 without classification head
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image: Image.Image) -> np.ndarray:
    """
    Extract a normalized 2048-dim embedding from a PIL image.
    """
    image = image.convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze().cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb.astype("float32")
