# backend/utils.py

from PIL import Image, ImageDraw
import torch
import numpy as np
import io
from torchvision import transforms

import os
import gdown

MODEL_PATH = "polygon_model.pth"
MODEL_DRIVE_ID = "1tGUienHqq33j7BLKv2_mK4q8fIFFnTgS"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded.")

def load_model(path):
    from backend.model import PolygonUNetDownClassifier
    model = PolygonUNetDownClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict_and_mask(image: Image.Image, model, resize=(128, 128)):
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        points = output.view(-1, 2).numpy()

    # Rescale to original image size
    scale_x = original_size[0] / resize[0]
    scale_y = original_size[1] / resize[1]
    points_original = np.array([[x * scale_x, y * scale_y] for x, y in points], dtype=np.float32)

    # Select points 1,6,7,8,9,10 → indexes 0,5,6,7,8,9
    selected_indices = [0, 5, 6, 7, 8, 9]
    selected_points = points_original[selected_indices]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sticker_path = os.path.join(BASE_DIR, "a.png")
    sticker = Image.open(sticker_path).convert("RGBA")

    # Convert base image to RGBA to support transparency
    base_image = image.convert("RGBA")

    # Get bounding box of selected polygon points
    min_x, min_y = np.min(selected_points, axis=0)
    max_x, max_y = np.max(selected_points, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    # Scale the sticker slightly ("stress" it)
    scale_factor = 1.01
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Center new sticker on polygon
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    paste_x = int(center_x - new_width / 2)
    paste_y = int(center_y - new_height / 2)

    # Resize sticker
    sticker_resized = sticker.resize((new_width, new_height), resample=Image.LANCZOS)

    # Transparent canvas to paste sticker
    sticker_canvas = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    sticker_canvas.paste(sticker_resized, (paste_x, paste_y), sticker_resized)

    # Composite sticker onto original image
    final_image = Image.alpha_composite(base_image, sticker_canvas)

    # Convert final result to PNG bytes
    buf = io.BytesIO()
    final_image.save(buf, format='PNG')
    buf.seek(0)
    return buf
