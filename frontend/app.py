import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Polygon Predictor", layout="centered")
st.title("Superheroes Face Masking")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    if st.button("Predict Mask"):
        # Send image to backend
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files={"file": files['file']})
        
        if response.status_code == 200:
            result_img = Image.open(io.BytesIO(response.content))

            # Show side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", width=300)
            with col2:
                st.image(result_img, caption="Masked Prediction", width=300)
        else:
            st.error("Prediction failed.")