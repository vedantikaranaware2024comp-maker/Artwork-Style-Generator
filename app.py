import streamlit as st
import tensorflow as tf
import numpy as np
import os
import urllib.request
import tarfile
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("🎨 AI Artwork Style Generator")
st.write("Upload a photograph and a painting to transfer the artistic style in seconds.")

# 2. Bulletproof Custom Model Loader
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed"
    
    # Create a visible folder named 'ai_style_model' in your current project directory
    model_dir = os.path.join(os.getcwd(), "ai_style_model")
    os.makedirs(model_dir, exist_ok=True)
    
    archive_path = os.path.join(model_dir, "model.tar.gz")
    
    # If the unzipped model doesn't exist yet, manually download and extract it
    if not os.path.exists(os.path.join(model_dir, "saved_model.pb")):
        with st.spinner("Downloading AI Model (This only happens once)..."):
            # Download with a browser User-Agent to bypass security blocks
            req = urllib.request.Request(model_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
                out_file.write(response.read())
            
            # Explicitly extract the archive
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=model_dir)
                
    # Load the cleanly extracted model
    model = tf.saved_model.load(model_dir)
    return model

model = load_model()

# 3. Image Processing Utility
def process_image(uploaded_file, max_dim=512):
    img = Image.open(uploaded_file).convert('RGB')
    img.thumbnail((max_dim, max_dim))
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype(np.float32) / 255.0
    img_tensor = tf.expand_dims(img_array, 0)
    
    return img_tensor, img

# 4. User Interface Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Content Image")
    content_file = st.file_uploader("Upload your photograph", type=["jpg", "jpeg", "png"], key="content")

with col2:
    st.subheader("2. Style Image")
    style_file = st.file_uploader("Upload reference artwork", type=["jpg", "jpeg", "png"], key="style")

# 5. Execution Logic
if content_file and style_file:
    content_tensor, content_pil = process_image(content_file)
    style_tensor, style_pil = process_image(style_file, max_dim=256) 
    
    col1.image(content_pil, use_container_width=True)
    col2.image(style_pil, use_container_width=True)
    
    st.markdown("---")
    
    if st.button("Generate Artwork 🚀", use_container_width=True):
        with st.spinner("Neural network is painting..."):
            
            # Pass images through the model
            results = model(tf.constant(content_tensor), tf.constant(style_tensor))
            stylized_tensor = results[0]
            
            stylized_image = np.array(stylized_tensor[0] * 255, dtype=np.uint8)
            
            st.subheader("Final Result")
            st.image(stylized_image, use_container_width=True)
            
            stylized_pil = Image.fromarray(stylized_image)
            import io
            buf = io.BytesIO()
            stylized_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Masterpiece",
                data=byte_im,
                file_name="stylized_artwork.jpg",
                mime="image/jpeg",
                use_container_width=True
            )