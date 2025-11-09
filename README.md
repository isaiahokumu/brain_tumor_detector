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
- **Deep Learning:** Classification & segmentation of brain tumors using CNNs.
- **Autoencoders:** Unsupervised anomaly detection to identify abnormal brain regions.
- **Web Interface:** Streamlit application for clinicians and researchers to visualize results.

---

## Project Architecture
```graphql
brain-tumor-detector/
│
├── eda_outputs/
│   ├── plots/
│   └── samples/
|
├── mri_dataset/
│   ├── Testing/              # Test data
|       ├── glioma
|       ├── meningioma
|       ├── notumor
|       └── pituitary
│   └── Training/             # Training data
|       ├── glioma
|       ├── meningioma
|       ├── notumor
|       └── pituitary
│
├── models/
│   ├── brain_tumor.keras         # CNN baseline model
│   └── autoencoder_model.keras   # Autoencoder anomaly detection
│
├── notebooks/
│   ├── code.ipynb
│   └── autoencoder.ipynb
│
├── src/
│   └── app.py                   # User interface
|
├── README.md
└── requirements.txt
```

---

## Features

| Component | Description |
|------------|-------------|
| **CNN Classifier** | Custom convolutional model for baseline classification |
| **Autoencoder** | Unsupervised anomaly detection using reconstruction errors |
| **Cloud Deployment** | Model hosting via Streamlit Cloud |

---

## Tech Stack

**Machine Learning**
- TensorFlow / Keras  
- Scikit-learn  
- NumPy / Pandas  
- OpenCV  

**Web**
- Streamlit
  
**Cloud** 
- Streamlit Cloud  

---

## Setup Instructions

### Clone the Repository

git clone[ https://github.com/isaiahokumu/brain-tumor-detector.git](https://github.com/isaiahokumu/brain_tumor_detector.git)

cd brain-tumor-detector


### Install Dependencies
pip install -r requirements.txt

### Download Dataset

Use the [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Unzip it into:
```bash
/data/raw/
```

### Train Models

Run notebooks in sequence:
```bash
jupyter notebook notebooks/code.ipynb
```
### Metrics
Model Metrics.

| Metric               | precision   | recall | f1score  | support |
| -------------------- | ----------- | -------| -------- | ------- |
| glioma               | 0.99        | 0.92   | 0.95     | 300     |
| meningioma           | 0.91        | 0.92   | 0.91     | 306     |
| notumor              | 0.96        | 0.92   | 0.97     | 405     |
| pituitary            | 0.87        | 0.92   | 0.98     | 300     |


**author**: Isaiah Okumu







