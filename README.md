# 🧠 Polygon Predictor Backend (FastAPI)

This is the backend for the Polygon Predictor app, built with FastAPI. It loads a pre-trained PyTorch model to apply polygon-style masks to uploaded images.

---

## 📁 Structure

```
backend/
├── main.py             # FastAPI app and /predict route
├── model.py            # Loads the model
├── utils.py            # Prediction and masking logic
├── polygon_model.pth   # Pre-trained model file (75MB)
├── a.png               # Mask overlay image
```

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 📦 Endpoint

### `POST /predict/`

- Accepts an image (`file`) and returns the masked image.
- Example:

```bash
curl -X POST http://localhost:8000/predict/ \
  -F "file=@your_image.png" --output result.png
```

---

## 🔗 Notes

- Download the model from [Google Drive](https://drive.google.com/file/d/1tGUienHqq33j7BLKv2_mK4q8fIFFnTgS/view?usp=sharing) and place it in the `backend/` directory.
- Ensure `a.png` is present in `backend/`, or use a full path in `utils.py`.

---
