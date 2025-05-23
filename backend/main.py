
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from utils import load_model, predict_and_mask, download_model
from PIL import Image
import io

app = FastAPI()
download_model()
model = load_model("polygon_model.pth")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    buf = predict_and_mask(image, model)
    return StreamingResponse(buf, media_type="image/png")
