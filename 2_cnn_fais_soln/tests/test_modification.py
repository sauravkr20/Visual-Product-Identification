
import numpy as np
from pathlib import Path
from PIL import Image
from PIL import Image, ImageFilter, ImageOps

def apply_modification(input_image_path, output_image_path):
    image = Image.open(input_image_path).convert("RGB")

    # 1. Slight scale down (95%)
    scale_factor = 0.85
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, resample=Image.LANCZOS)

    # 2. Rotate 30 degrees clockwise
    # PIL rotates counter-clockwise by default, so use -30 degrees for clockwise
    image = image.rotate( 4, resample=Image.BICUBIC, expand=False)

    # 3. Convert to sepia
    np_img = np.array(image)
    tr = [0.393, 0.769, 0.189]
    tg = [0.349, 0.686, 0.168]
    tb = [0.272, 0.534, 0.131]

    r = np_img[:, :, 0]
    g = np_img[:, :, 1]
    b = np_img[:, :, 2]

    sepia_r = (r * tr[0] + g * tr[1] + b * tr[2]).clip(0, 255)
    sepia_g = (r * tg[0] + g * tg[1] + b * tg[2]).clip(0, 255)
    sepia_b = (r * tb[0] + g * tb[1] + b * tb[2]).clip(0, 255)

    sepia_img = np.stack([sepia_r, sepia_g, sepia_b], axis=2).astype(np.uint8)
    image = Image.fromarray(sepia_img)

    # 4. Slight Gaussian blur
    # image = image.filter(ImageFilter.GaussianBlur(radius=1))

    # Save modified image
    image.save(output_image_path)

