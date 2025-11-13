import streamlit as st
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="Brain MRI Analyzer",
    layout="wide",
    page_icon="🧠",
)

# CONFIGURATION
MAX_FILE_SIZE_MB = 10
MODEL_PATH = "brain_tumor.keras"
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# LOAD MODEL WITH ERROR HANDLING
@st.cache_resource
def load_brain_model():
    try:
        if not Path(MODEL_PATH).exists():
            st.error(f"Model file not found: {MODEL_PATH}")
            return None
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_brain_model()

# HELPER FUNCTIONS
def validate_file_size(uploaded_file):
    """Check if file size is within limits"""
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
        return False
    return True

def preprocess_image(img):
    """Preprocess image for model input"""
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



# SIDEBAR MENU
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose section:", ["Home", "MRI Analysis"])

# Status indicators
with st.sidebar:
    st.divider()
    st.caption("**System Status**")
    
    # MRI Model Status
    if model:
        st.success("✓ MRI Model Loaded")
    else:
        st.error("✗ MRI Model Not Available")

# HOME
if option == "Home":
    st.title("Brain MRI Analyzer")
    st.markdown(f"""
    Welcome to the **AI-driven Brain Tumor Detection System**.

    ### Features
    - **MRI Scan Analysis** → Upload brain MRI images to detect tumor type
    - **Multi-class Classification** → Identifies 4 different conditions
    - **Batch Processing** → Analyze multiple images at once
    - **Downloadable Predictions** → Export results as CSV

    ### How It Works
    This tool uses a **Convolutional Neural Network (CNN)** trained to classify brain MRI scans into four categories:
    
    1. **Glioma** - Tumors arising from glial cells
    2. **Meningioma** - Tumors of the meninges (protective membranes)
    3. **Pituitary** - Tumors of the pituitary gland
    4. **No Tumor** - Normal brain tissue
    
    The model analyzes the MRI image through multiple layers of feature extraction to identify patterns associated with each tumor type.

    ### Usage Instructions
    1. Navigate to **MRI Analysis** in the sidebar
    2. Upload one or more MRI images (JPG, JPEG, or PNG)
    3. View predictions with confidence scores
    4. Download results as CSV for record-keeping

    ---
    ### Important Disclaimers
    - This tool is for **research and educational purposes only**
    - Not intended for clinical diagnosis or medical decision-making
    - Always consult qualified healthcare professionals for medical advice
    - Maximum file size: {MAX_FILE_SIZE_MB}MB per file
    - Supported formats: JPG, JPEG, PNG

    ### About the Technology
    The underlying CNN architecture uses:
    - **3 Convolutional Blocks** for hierarchical feature learning
    - **MaxPooling** for spatial downsampling
    - **Dropout & L2 Regularization** to prevent overfitting
    
    ---
    **Ready to get started?** Click on "MRI Analysis" in the sidebar!
    """)

# MRI ANALYSIS
elif option == "MRI Analysis":
    st.header("Brain MRI Image Classification")

    if model is None:
        st.error("MRI classification model not available. Please check model file.")
        st.stop()

    st.markdown("""
    Upload one or multiple MRI brain scans for automatic tumor detection and classification.
    The model will analyze each image and provide confidence scores for all tumor types.
    """)

    uploaded_files = st.file_uploader(
        "Upload MRI image(s)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILE_SIZE_MB}MB per file. You can select multiple images."
    )
    
    if uploaded_files:
        results = []
        
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            if not validate_file_size(uploaded_file):
                continue
                
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            try:
                img = Image.open(uploaded_file).convert("RGB")
                img_array = preprocess_image(img)
                
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    prediction = model.predict(img_array, verbose=0)
                    confidence = np.max(prediction)
                    pred_class = class_names[np.argmax(prediction)]

                with col1:
                    st.subheader(f"Result #{idx + 1}")
                    st.image(img, caption=uploaded_file.name, use_container_width=True)

                with col2:
                    st.subheader(f"Analysis")
                    st.metric("Prediction", pred_class)
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    with st.expander("View all class probabilities"):
                        for i, class_name in enumerate(class_names):
                            prob = prediction[0][i] * 100
                            st.progress(float(prediction[0][i]), text=f"{class_name}: {prob:.2f}%")

                results.append({
                    "Filename": uploaded_file.name,
                    "Prediction": pred_class,
                    "Confidence": f"{confidence:.4f}",
                    **{f"{class_name}_prob": f"{prediction[0][i]:.4f}" for i, class_name in enumerate(class_names)}
                })
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        if results:
            st.divider()
            st.subheader("Summary & Export")
            df = pd.DataFrame(results)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df, use_container_width=True)
            with col2:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV", 
                    data=csv, 
                    file_name="mri_predictions.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.metric("Total Images", len(results))
                
            st.success(f"✓ Successfully processed {len(results)} image(s)")
    else:
        st.info("Upload MRI images above to begin analysis")
        
        with st.expander("Tips for best results"):
            st.markdown("""
            - Use high-quality MRI scans
            - Ensure images are properly oriented
            - Axial (horizontal) slices work best
            - Supported formats: JPG, JPEG, PNG
            - Maximum file size: 10MB per image
            - You can upload multiple images at once
            """)

# Footer
st.sidebar.divider()
st.sidebar.caption("For educational use only")
st.sidebar.caption("Not for clinical diagnosis")