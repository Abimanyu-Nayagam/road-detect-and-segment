##Road Detection and Segmentation System
Overview
This project implements a two-stage deep learning system for detecting and segmenting road areas in images and videos.
It is designed as a component of a larger self-driving car system, helping distinguish drivable surfaces from non-drivable regions.

The pipeline includes:

Stage 1: Anomaly Detection — An autoencoder model determines if the input frame is a valid road scene.

Stage 2: Semantic Segmentation — A modified U-Net model classifies each pixel as "road" or "non-road," provided the first stage gives a positive result.

Live inference is available through the hosted link.

##Key Components
1. Anomaly Detection (Autoencoder)
Input: 2048-dimensional features extracted via a pretrained ResNet-50.

Feature Reduction: PCA to reduce dimensionality from 2048 → 512.

Output: Anomaly score — decides whether segmentation should proceed.

2. Semantic Segmentation (Modified U-Net)
Input: Raw image/frame.

Output: Pixel-wise segmentation mask distinguishing road vs. non-road.

Datasets
Road Segmentation:

Trained using the Cityscapes Dataset.

Custom masks were created by simplifying multi-class labels into binary "road" vs "non-road" masks.

Anomaly Detection:

Custom road-only dataset and autoencoder trained the Kaggle dataset: 100k Vehicle Dashcam Image Dataset. https://www.kaggle.com/datasets/mdfahimbinamin/100k-vehicle-dashcam-image-dataset

##How to Run

Install dependencies
```
pip install -r requirements.txt
```
Run the Streamlit App
```
streamlit run app/streamlit_app.py
```
Upload a video or image

The app will first perform anomaly detection.

If the input passes, pixel-wise segmentation will be shown.
