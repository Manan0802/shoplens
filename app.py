import streamlit as st
from PIL import Image

# Image preprocessing utilities
from utils.image_utils import pil_to_numpy, resize_image, normalize_image

# CLIP model utilities
from models.clip_model import load_clip_model, generate_image_embedding


# -----------------------------
# Load CLIP Model (once)
# -----------------------------
model, processor = load_clip_model()


# -----------------------------
# App Title
# -----------------------------
st.title("🛒 ShopLens - Visual Search Engine")

st.write("Upload a clothing image to generate its visual embedding.")


# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# When Image is Uploaded
# -----------------------------
if uploaded_file is not None:

    # Load image using PIL
    pil_image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(pil_image, use_container_width=True)


    # -----------------------------
    # Preprocessing
    # -----------------------------

    numpy_image = pil_to_numpy(pil_image)

    resized_image = resize_image(numpy_image)

    normalized_image = normalize_image(resized_image)


    st.subheader("Processed Image")
    st.image(resized_image, use_container_width=True)


    # -----------------------------
    # Generate CLIP Embedding
    # -----------------------------

    embedding = generate_image_embedding(pil_image, model, processor)


    # -----------------------------
    # Display Embedding Info
    # -----------------------------

    st.subheader("Embedding Information")

    st.write("Embedding Dimension:", len(embedding))

    st.write("First 10 values of the embedding vector:")

    st.write(embedding[:10])