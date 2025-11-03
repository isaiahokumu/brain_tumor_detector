import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from io import BytesIO
import os, zipfile, tempfile
from PIL import Image

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection Dashboard", layout="wide")
st.title("Brain Tumor Detection using Deep Learning")

# Class labels (adjust for your model)
class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_CNN_model(path):
    return load_model(path)

model_path = "brain_tumor.keras"
if not os.path.exists(model_path):
    st.warning("No model found. Please upload a trained model (.h5 or .keras)")
    uploaded_model = st.file_uploader("Upload model", type=["h5", "keras"])
    if uploaded_model:
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
else:
    st.sidebar.success("Model loaded successfully.")

cnn_model = load_CNN_model(model_path)
input_shape = cnn_model.input_shape[1:3]
channels = cnn_model.input_shape[-1]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_image(img, target_size=input_shape, color_mode="rgb"):
    if isinstance(img, str):
        img = image.load_img(img, target_size=target_size, color_mode=color_mode)
    elif isinstance(img, BytesIO):
        img = Image.open(img).convert("RGB").resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    if channels == 1 and img_array.ndim == 3 and img_array.shape[-1] == 3:
        img_array = tf.image.rgb_to_grayscale(img_array)
    return np.expand_dims(img_array, axis=0), img

def predict_image(img_array):
    preds = cnn_model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])
    return class_labels[pred_idx], confidence, preds[0]

# --- GRAD-CAM ---
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0] * pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_gradcam(heatmap, original_img, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(original_img.size)
    heatmap = np.array(heatmap.convert("RGB"))
    overlay = np.array(original_img) * (1 - alpha) + heatmap * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))
    return overlay

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.radio(
    "Navigate App",
    ["How to Use", "Single Prediction", "Batch Prediction", "Performance Dashboard"]
)

# =====================================================
# HOW TO USE TAB
# =====================================================
if menu == "How to Use":
    st.header("How to Use This App")

    st.markdown("""
    ### Steps:
    1. **Upload one MRI** under “Single Prediction” to see classification and Grad-CAM focus.
    2. Or use “Batch Prediction” to upload a ZIP of images for mass predictions.
    3. Export predictions as CSV including Grad-CAM overlay paths.
    4. View evaluation metrics in “Performance Dashboard”.

    ---
    ### Example MRI Samples
    """)
    cols = st.columns(4)
    samples = {
        "Glioma": "samples/glioma.jpg",
        "Meningioma": "samples/meningioma.jpg",
        "Pituitary": "samples/pituitary.jpg",
        "No Tumor": "samples/normal.jpg",
    }
    for i, (label, path) in enumerate(samples.items()):
        with cols[i]:
            if os.path.exists(path):
                st.image(path, caption=label, use_container_width=True)
            else:
                st.info(f"Upload sample: {path}")

# =====================================================
# SINGLE IMAGE PREDICTION
# =====================================================
elif menu == "Single Prediction":
    st.header("Upload an MRI Image for Prediction")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_array, img = preprocess_image(uploaded_file)
        label, conf, probs = predict_image(img_array)

        st.image(uploaded_file, caption=f"Prediction: {label} ({conf*100:.2f}%)", use_container_width=True)
        st.bar_chart(dict(zip(class_labels, probs)))

        if st.button("Generate Grad-CAM"):
            try:
                # Find last conv layer
                last_conv_layer = None
                for layer in reversed(cnn_model.layers):
                    if len(layer.output_shape) == 4:
                        last_conv_layer = layer.name
                        break
                if not last_conv_layer:
                    st.warning("No convolutional layer found for Grad-CAM.")
                else:
                    heatmap = grad_cam(cnn_model, img_array, last_conv_layer)
                    overlay = overlay_gradcam(heatmap, img)
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

# =====================================================
# BATCH PREDICTION
# =====================================================
elif menu == "Batch Prediction":
    st.header("Upload ZIP of MRI Images")
    uploaded_zip = st.file_uploader("Upload ZIP", type=["zip"])
    if uploaded_zip:
        temp_dir = tempfile.mkdtemp()
        gradcam_dir = os.path.join(temp_dir, "gradcam_outputs")
        os.makedirs(gradcam_dir, exist_ok=True)

        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        image_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        results = []
        for img_path in image_files:
            img_array, img = preprocess_image(img_path)
            label, conf, _ = predict_image(img_array)
            # Grad-CAM overlay
            try:
                for layer in reversed(cnn_model.layers):
                    if len(layer.output_shape) == 4:
                        last_conv = layer.name
                        break
                heatmap = grad_cam(cnn_model, img_array, last_conv)
                overlay = overlay_gradcam(heatmap, img)
                overlay_path = os.path.join(gradcam_dir, f"{os.path.basename(img_path)}_gradcam.jpg")
                Image.fromarray(overlay).save(overlay_path)
            except Exception:
                overlay_path = "N/A"
            results.append({
                "Filename": os.path.basename(img_path),
                "Prediction": label,
                "Confidence": round(conf * 100, 2),
                "GradCAM_Path": overlay_path
            })

        df = pd.DataFrame(results)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results (CSV)", data=csv,
                           file_name="batch_predictions.csv", mime="text/csv")

# =====================================================
# PERFORMANCE DASHBOARD
# =====================================================
elif menu == "Performance Dashboard":
    st.header("Model Performance Dashboard")

    test_path = st.text_input("Enter test dataset directory:", "mri_dataset/Testing")
    if st.button("Run Evaluation"):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_gen = datagen.flow_from_directory(
            test_path,
            target_size=input_shape,
            color_mode="rgb" if channels == 3 else "grayscale",
            class_mode="categorical",
            shuffle=False
        )

        y_true = test_gen.classes
        y_pred_probs = cnn_model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
        st.metric("Weighted F2 Score", f"{f2:.4f}")

        report = classification_report(y_true, y_pred,
                                       target_names=list(test_gen.class_indices.keys()),
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(cm, text_auto=True, x=class_labels, y=class_labels, color_continuous_scale="Blues")
        st.plotly_chart(fig)
