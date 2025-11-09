import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from pathlib import Path
import json
import requests

# 🔹 PAGE CONFIG
st.set_page_config(
    page_title="Brain MRI Analyzer",
    layout="wide",
    page_icon="🧠",
)

# 🔹 CONFIGURATION
MAX_FILE_SIZE_MB = 10
MODEL_PATH = "brain_tumor.keras"
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# Initialize session state for API settings
if 'api_provider' not in st.session_state:
    st.session_state.api_provider = "OpenAI"
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = os.environ.get('OPENAI_API_KEY', '')
if 'anthropic_key' not in st.session_state:
    st.session_state.anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = os.environ.get('GEMINI_API_KEY', '')

# 🔹 LOAD MODEL WITH ERROR HANDLING
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

# 🔹 API FUNCTIONS

def ask_openai(question, context, api_key):
    """Use OpenAI API for clinical note Q&A"""
    try:
        if not api_key:
            return "API Error: No OpenAI API key provided."
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant analyzing clinical notes. Provide clear, concise answers based ONLY on the information in the clinical note provided. If the answer cannot be found in the note, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": f"""Clinical Note:
{context}

Question: {question}"""
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        
        return "Unable to generate answer."
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "API Error: Invalid OpenAI API key."
        elif e.response.status_code == 429:
            return "API Error: Rate limit exceeded or insufficient quota."
        else:
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
                return f"API Error: {error_detail}"
            except:
                return f"API Error: {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def ask_anthropic(question, context, api_key):
    """Use Anthropic Claude API for clinical note Q&A"""
    try:
        if not api_key:
            return "API Error: No Anthropic API key provided."
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": f"""You are a medical AI assistant analyzing clinical notes. 

Clinical Note:
{context}

Question: {question}

Please provide a clear, concise answer based ONLY on the information in the clinical note above. If the answer cannot be found in the note, say so clearly."""
                }]
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        answer_text = ""
        if "content" in data:
            for item in data["content"]:
                if item.get("type") == "text":
                    answer_text += item.get("text", "")
        
        return answer_text if answer_text else "Unable to generate answer."
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "API Error: Invalid Anthropic API key."
        elif e.response.status_code == 429:
            return "API Error: Rate limit exceeded."
        else:
            return f"API Error: {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def ask_gemini(question, context, api_key):
    """Use Google Gemini API for clinical note Q&A"""
    try:
        if not api_key:
            return "API Error: No Gemini API key provided."
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "contents": [{
                    "parts": [{
                        "text": f"""You are a medical AI assistant analyzing clinical notes. Provide clear, concise answers based ONLY on the information in the clinical note provided. If the answer cannot be found in the note, say so clearly.

Clinical Note:
{context}

Question: {question}"""
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1000
                }
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if "candidates" in data and len(data["candidates"]) > 0:
            content = data["candidates"][0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "Unable to generate answer.")
        
        return "Unable to generate answer."
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", "")
                if "API key" in error_msg:
                    return "API Error: Invalid Gemini API key."
                return f"API Error: {error_msg}"
            except:
                return "API Error: Invalid request or API key."
        elif e.response.status_code == 429:
            return "API Error: Rate limit exceeded."
        else:
            return f"API Error: {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def ask_ai(question, context):
    """Route to the appropriate AI provider"""
    provider = st.session_state.api_provider
    
    if provider == "OpenAI":
        return ask_openai(question, context, st.session_state.openai_key)
    elif provider == "Anthropic":
        return ask_anthropic(question, context, st.session_state.anthropic_key)
    elif provider == "Gemini":
        return ask_gemini(question, context, st.session_state.gemini_key)
    else:
        return "Error: Unknown API provider selected."

def get_current_api_key():
    """Get the API key for the current provider"""
    provider = st.session_state.api_provider
    if provider == "OpenAI":
        return st.session_state.openai_key
    elif provider == "Anthropic":
        return st.session_state.anthropic_key
    elif provider == "Gemini":
        return st.session_state.gemini_key
    return ""

# 🔹 HELPER FUNCTIONS
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

def find_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def generate_gradcam(model, img_array, layer_name=None):
    """Generate Grad-CAM heatmap"""
    try:
        if layer_name is None:
            layer_name = find_last_conv_layer(model)
            if layer_name is None:
                raise ValueError("No convolutional layer found in model")
        
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception as e:
        raise Exception(f"Grad-CAM generation failed: {str(e)}")

def overlay_gradcam(heatmap, img):
    """Overlay Grad-CAM heatmap on original image"""
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
    return overlay

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF with error handling"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    except ImportError:
        st.error("PyPDF2 not installed. Run: pip install PyPDF2")
        return None
    except Exception as e:
        st.error(f"Failed to extract PDF text: {str(e)}")
        return None

def truncate_context(text, max_length=50000):
    """Truncate long text to fit model context window"""
    if len(text) > max_length:
        st.warning(f"Note truncated to {max_length} characters for processing")
        return text[:max_length]
    return text

# 🔹 SIDEBAR MENU
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose section:", ["Home", "MRI Analysis", "Clinical Notes Assistant"])

# API Configuration
st.sidebar.divider()
st.sidebar.subheader("API Configuration")

# API Provider Selection
api_provider = st.sidebar.selectbox(
    "Select AI Provider",
    ["OpenAI", "Anthropic", "Gemini"],
    index=["OpenAI", "Anthropic", "Gemini"].index(st.session_state.api_provider),
    help="Choose which AI provider to use for clinical note analysis"
)
st.session_state.api_provider = api_provider

# Show configuration for selected provider
with st.sidebar.expander(f"{api_provider} API Settings", expanded=not get_current_api_key()):
    if api_provider == "OpenAI":
        st.markdown("""
        **GPT-4o** - Latest OpenAI model
        
        Get your API key at: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        
        **Cost:** ~$0.005 per analysis
        """)
        
        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_key,
            type="password",
            key="openai_input"
        )
        
        if st.button("Save OpenAI Key", key="save_openai"):
            st.session_state.openai_key = openai_key_input
            if openai_key_input:
                st.success("✓ OpenAI key saved!")
            else:
                st.warning("Key cleared")
    
    elif api_provider == "Anthropic":
        st.markdown("""
        **Claude Sonnet 4** - Advanced reasoning
        
        Get your API key at: [console.anthropic.com](https://console.anthropic.com/)
        
        **Cost:** ~$0.003 per analysis
        """)
        
        anthropic_key_input = st.text_input(
            "Anthropic API Key",
            value=st.session_state.anthropic_key,
            type="password",
            key="anthropic_input"
        )
        
        if st.button("Save Anthropic Key", key="save_anthropic"):
            st.session_state.anthropic_key = anthropic_key_input
            if anthropic_key_input:
                st.success("✓ Anthropic key saved!")
            else:
                st.warning("Key cleared")
    
    elif api_provider == "Gemini":
        st.markdown("""
        **Gemini Pro** - Google's AI model
        
        Get your API key at: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
        
        **Cost:** Free tier available!
        """)
        
        gemini_key_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_key,
            type="password",
            key="gemini_input"
        )
        
        if st.button("Save Gemini Key", key="save_gemini"):
            st.session_state.gemini_key = gemini_key_input
            if gemini_key_input:
                st.success("✓ Gemini key saved!")
            else:
                st.warning("Key cleared")

# Status indicators
with st.sidebar:
    st.divider()
    st.caption("**System Status**")
    
    # MRI Model Status
    if model:
        st.success("✓ MRI Model Loaded")
    else:
        st.error("✗ MRI Model Not Available")
    
    # API Status
    current_key = get_current_api_key()
    if current_key:
        st.success(f"✓ {api_provider} API Configured")
    else:
        st.warning(f"⚠ {api_provider} API Key Missing")
    
    # Show all configured providers
    with st.expander("All API Keys Status"):
        st.caption("**OpenAI:**")
        st.write("✓ Configured" if st.session_state.openai_key else "✗ Not configured")
        
        st.caption("**Anthropic:**")
        st.write("✓ Configured" if st.session_state.anthropic_key else "✗ Not configured")
        
        st.caption("**Gemini:**")
        st.write("✓ Configured" if st.session_state.gemini_key else "✗ Not configured")

# HOME
if option == "Home":
    st.title("Brain MRI Analyzer + Clinical Assistant")
    st.markdown(f"""
    Welcome to the integrated **AI-driven Brain Tumor Diagnostic Assistant**.

    ### Features
    - **MRI Scan Analysis** → Upload brain MRI images to detect tumor type (No API key required)
    - **Clinical Notes Assistant** → Powered by your choice of AI:
      - **OpenAI GPT-4o** - Latest and most capable
      - **Anthropic Claude** - Advanced reasoning
      - **Google Gemini** - Free tier available
    - **Grad-CAM Visualization** → See what regions the model focuses on  
    - **Downloadable Predictions** → Export your model predictions as CSV

    ### Current AI Provider: **{st.session_state.api_provider}**
    
    Setup your preferred AI provider in the sidebar to enable Clinical Notes Assistant.
    
    **Note:** MRI Analysis works immediately without any API key.

    ### Important Disclaimers
    - This tool is for **research and educational purposes only**
    - Not intended for clinical diagnosis
    - Always consult qualified healthcare professionals
    - Maximum file size: {MAX_FILE_SIZE_MB}MB per file

    ---
    ### Supported Tumor Types
    - **Glioma** - Tumors arising from glial cells
    - **Meningioma** - Tumors of the meninges
    - **Pituitary** - Tumors of the pituitary gland
    - **No Tumor** - Normal brain tissue
    """)

# MRI ANALYSIS
elif option == "MRI Analysis":
    st.header("Brain MRI Image Classification")

    if model is None:
        st.error("MRI classification model not available. Please check model file.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload MRI image(s)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help=f"Maximum {MAX_FILE_SIZE_MB}MB per file"
    )
    
    if uploaded_files:
        results = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            if not validate_file_size(uploaded_file):
                continue
                
            st.divider()
            col1, col2 = st.columns(2)
            
            try:
                img = Image.open(uploaded_file).convert("RGB")
                img_array = preprocess_image(img)
                
                with st.spinner(f"Analyzing {uploaded_file.name}..."):
                    prediction = model.predict(img_array, verbose=0)
                    confidence = np.max(prediction)
                    pred_class = class_names[np.argmax(prediction)]

                with col1:
                    st.subheader(f"Result #{idx + 1}")
                    st.metric("Prediction", pred_class)
                    st.metric("Confidence", f"{confidence*100:.2f}%")
                    st.image(img, caption=uploaded_file.name, use_container_width=True)
                    
                    with st.expander("View all predictions"):
                        for i, class_name in enumerate(class_names):
                            st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

                with col2:
                    try:
                        heatmap = generate_gradcam(model, img_array)
                        overlay = overlay_gradcam(heatmap, img)
                        st.subheader("Grad-CAM Visualization")
                        st.image(overlay, caption="Model Attention Map", use_container_width=True)
                        st.caption("Warmer colors indicate regions the model focused on")
                    except Exception as e:
                        st.warning(f"Grad-CAM visualization unavailable: {str(e)}")

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
            st.subheader("Export Results")
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions (CSV)", 
                data=csv, 
                file_name="mri_predictions.csv", 
                mime="text/csv"
            )
            st.dataframe(df, use_container_width=True)

# CLINICAL NOTES ASSISTANT
elif option == "Clinical Notes Assistant":
    st.header("Medical Notes Analysis Chatbot")
    st.caption(f"Powered by {st.session_state.api_provider}")

    # Check API key status
    current_key = get_current_api_key()
    if not current_key:
        st.warning(f"No {st.session_state.api_provider} API key configured")
        
        # Provider-specific setup instructions
        if st.session_state.api_provider == "OpenAI":
            st.info("""
            **OpenAI Setup:**
            1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
            2. Create an account or sign in
            3. Click "Create new secret key"
            4. Copy your API key and enter it in the sidebar
            
            **Cost:** ~$0.005 per analysis (very affordable)
            """)
        elif st.session_state.api_provider == "Anthropic":
            st.info("""
            **Anthropic Setup:**
            1. Go to [console.anthropic.com](https://console.anthropic.com/)
            2. Create an account or sign in
            3. Navigate to API Keys section
            4. Create a new API key and enter it in the sidebar
            
            **Cost:** ~$0.003 per analysis
            """)
        elif st.session_state.api_provider == "Gemini":
            st.info("""
            **Gemini Setup:**
            1. Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy your API key and enter it in the sidebar
            
            **Cost:** FREE tier available! 🎉
            """)
        
        st.info("You can switch between providers in the sidebar at any time.")
        st.stop()

    st.markdown("""
    Upload or paste a clinical note and ask medical questions about it.
    
    **Tip**: Can handle documents up to 50,000 characters.
    """)

    uploaded_note = st.file_uploader(
        "Upload clinical note (.txt or .pdf)", 
        type=["txt", "pdf"],
        help=f"Maximum {MAX_FILE_SIZE_MB}MB"
    )

    note_text = ""
    if uploaded_note:
        if not validate_file_size(uploaded_note):
            st.stop()
            
        if uploaded_note.name.endswith(".txt"):
            note_text = uploaded_note.read().decode("utf-8")
        elif uploaded_note.name.endswith(".pdf"):
            note_text = extract_text_from_pdf(uploaded_note)
            if note_text is None:
                st.stop()

    note_input = st.text_area(
        "Or paste your clinical notes here:", 
        value=note_text, 
        height=200,
        placeholder="Enter clinical notes, patient history, lab results, imaging findings, etc."
    )

    if note_input:
        st.info(f"Note length: {len(note_input):,} characters")

    with st.expander("Example Questions"):
        st.markdown("""
        - What were the patient's presenting symptoms?
        - What diagnostic tests were performed?
        - What medications were prescribed?
        - What is the primary diagnosis?
        - Are there any abnormal lab values?
        - What follow-up care was recommended?
        """)

    question = st.text_input(
        "Ask a question about this note:",
        placeholder="e.g., What were the patient's symptoms?"
    )
    
    if st.button("Analyze Note", type="primary") and note_input and question:
        try:
            with st.spinner(f"{st.session_state.api_provider} is analyzing the clinical note..."):
                context = truncate_context(note_input)
                
                answer = ask_ai(question, context)
                
                if answer and not answer.startswith("Error:") and not answer.startswith("API Error:"):
                    st.success("**Answer:**")
                    st.markdown(answer)
                else:
                    st.error(answer)
                    if "API key" in answer or "Invalid" in answer:
                        st.info(f"Please check your {st.session_state.api_provider} API key in the sidebar.")
                    
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

    if not note_input:
        st.info("Upload or paste clinical notes above to start analysis.")
        
        with st.expander("See a demo example"):
            demo_note = """
PATIENT: John Doe, 45M
DATE: November 10, 2025

CHIEF COMPLAINT: Severe headaches and vision problems

HISTORY OF PRESENT ILLNESS:
Patient presents with progressive headaches over the past 3 months, accompanied by 
blurred vision in the left eye. Headaches are described as throbbing, worse in the 
morning, rated 8/10 in severity.

PHYSICAL EXAMINATION:
- BP: 145/90 mmHg
- Visual field defect in left temporal field
- Papilledema noted on fundoscopic exam

IMAGING:
MRI Brain with contrast reveals 3.2 cm enhancing mass in the pituitary region, 
consistent with pituitary macroadenoma with suprasellar extension.

ASSESSMENT: Pituitary macroadenoma with optic chiasm compression

PLAN:
1. Refer to neurosurgery for transsphenoidal resection
2. Endocrinology consult for hormone level assessment
3. Start dexamethasone 4mg q6h
4. Follow-up in 1 week
            """
            st.text_area("Demo Clinical Note:", demo_note, height=300, disabled=True)
            st.caption("Try asking: 'What imaging findings were reported?' or 'What is the treatment plan?'")

# Footer
st.sidebar.divider()
st.sidebar.caption("For educational use only. Not for clinical diagnosis.")      