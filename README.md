# Brain Tumor Detection & Intelligence System

An integrated deep learning project that detects brain tumors from MRI scans using **Convolutional Neural Networks (CNNs)**, **VGG16/VGG19 transfer learning**, and **Autoencoders** for anomaly detection and **cloud-based deployment**.

---

## Project Overview

This project aims to build an end-to-end AI-powered system for **brain tumor detection, research awareness, and insight generation**.

## Problem Statement
The accurate and timely diagnosis of brain tumors is a critical challenge in modern medicine. Manual analysis of Magnetic Resonance Imaging (MRI) scans by radiologists is a meticulous and time-consuming process that is prone to human error, potentially leading to misdiagnosis or delayed treatment. The subtle variations in tumor morphology, size, and location make it difficult to distinguish between different tumor types, such as gliomas, meningiomas, and pituitary tumors. This diagnostic bottleneck can severely impact patient outcomes, as the success of treatment often depends on early and precise intervention.

While automated systems have been proposed, many existing solutions suffer from limitations, including a high dependency on hand-crafted features, which fail to capture the complex, underlying patterns in medical images. Furthermore, these systems often lack a comprehensive framework that integrates diverse data sources, such as unstructured text from medical reports and external research news, into the diagnostic workflow.

This project addresses these challenges by proposing a robust and integrated deep learning system for the automated detection and classification of brain tumors from MRI scans. The primary objective is to develop a highly accurate and efficient model that can differentiate between various tumor types and normal brain tissue.

### Key Components
- **Deep Learning:** Classification & segmentation of brain tumors using CNNs and transfer models (VGG16/VGG19).
- **Autoencoders:** Unsupervised anomaly detection to identify abnormal brain regions.
- **Web Interface:** Streamlit application for clinicians and researchers to visualize results.

---

## Project Architecture
```graphql
brain-tumor-ai/
│
├── mri_dataset/
│   ├── raw/                 # Raw MRI data
│   ├── processed/           # Preprocessed/normalized data
│   └── reports/             # OCR input files
│
├── models/
│   ├── cnn_model.h5         # CNN baseline model
│   ├── vgg16_model.h5       # Transfer learning model (VGG16)
│   ├── vgg19_model.h5       # Transfer learning model (VGG19)
│   └── autoencoder_model.h5 # Autoencoder anomaly detection
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_cnn_training.ipynb
│   ├── 03_vgg16_training.ipynb
│   ├── 04_autoencoder.ipynb
│   ├── 05_gradcam_visualization.ipynb
│   └── 06_sentiment_analysis.ipynb
│
├── src/
│   ├── data_loader.py           # Handles data ingestion and augmentation
│   ├── model_cnn.py             # CNN architecture
│   └── app.py                   # User interface
│
├── tests/
│   ├── test_models.py
│   ├── test_ocr.py
│   └── test_api.py
│
├── README.md
└── LICENSE
```

---

## Features

| Component | Description |
|------------|-------------|
| **CNN Classifier** | Custom convolutional model for baseline classification |
| **Transfer Learning (VGG16/VGG19)** | Pre-trained models fine-tuned for tumor detection |
| **Autoencoder** | Unsupervised anomaly detection using reconstruction errors |
| **Cloud Deployment** | Model hosting via Streamlit Cloud |
| **Explainability (Grad-CAM)** | Visual heatmaps showing regions influencing predictions |

---

## Tech Stack

**Machine Learning**
- TensorFlow / Keras  
- Scikit-learn  
- NumPy / Pandas  
- OpenCV  

**NLP & OCR**
- Tesseract (OCR)  
- Hugging Face Transformers  
- BeautifulSoup / Requests  

**Web**
- Streamlit
  
**Cloud** 
- Streamlit Cloud  

---

## Setup Instructions

### Clone the Repository

git clone https://github.com/<your-username>/brain-tumor-ai.git
cd brain-tumor-ai


### Install Dependencies
pip install -r deployment/requirements.txt

### Download Dataset

Use the Kaggle Brain MRI Dataset

Unzip it into:
```bash
/data/raw/
```

### Train Models

Run notebooks in sequence:
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```
### Cloud Deployment
Use GCP Vertex AI for training and prediction.

| Metric               | CNN | VGG16 | VGG19 |
| -------------------- | --- | ----- | ----- |
| Accuracy             | TBD | TBD   | TBD   |
| Recall (Sensitivity) | TBD | TBD   | TBD   |
| F1 Score             | TBD | TBD   | TBD   |
| AUC                  | TBD | TBD   | TBD   |


