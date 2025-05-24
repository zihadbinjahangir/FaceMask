# backend/utils.py

from PIL import Image, ImageDraw
import torch
import numpy as np
import io
from torchvision import transforms

import os
import gdown
import cv2

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

def predict_and_mask(full_image_pil: Image.Image, model, sticker_path="a.png", resize=(128, 128), device='cpu'):
    # Convert PIL to OpenCV image for face detection
    image_cv = cv2.cvtColor(np.array(full_image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected.")
        return None

    x, y, w, h = faces[0]
    face_crop = image_cv[y:y+h, x:x+w]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
    original_face_size = face_pil.size

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(face_tensor)
        points = output.view(-1, 2).cpu().numpy()

    # Rescale predicted points to original face crop size
    scale_x = original_face_size[0] / resize[0]
    scale_y = original_face_size[1] / resize[1]
    points_original = np.array([[px * scale_x, py * scale_y] for px, py in points], dtype=np.float32)

    selected_indices = [0, 5, 6, 7, 8, 9]
    selected_points = np.array([points_original[i] for i in selected_indices])

    # Get bounding box from selected polygon points
    min_x, min_y = np.min(selected_points, axis=0)
    max_x, max_y = np.max(selected_points, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    # Scale and center sticker
    scale_factor = 1.05
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    sticker_x = int(center_x - new_width / 2)
    sticker_y = int(center_y - new_height / 2)

    # Load and resize sticker
    sticker = Image.open(sticker_path).convert("RGBA")
    resized_sticker = sticker.resize((new_width, new_height), resample=Image.LANCZOS)

    # Composite sticker onto transparent canvas at correct location
    sticker_layer = Image.new("RGBA", full_image_pil.size, (0, 0, 0, 0))
    sticker_layer.paste(resized_sticker, (x + sticker_x, y + sticker_y), resized_sticker)

    final_image = Image.alpha_composite(full_image_pil.convert("RGBA"), sticker_layer)

    # Return image as byte stream
    buf = io.BytesIO()
    final_image.save(buf, format='PNG')
    buf.seek(0)
    return buf
