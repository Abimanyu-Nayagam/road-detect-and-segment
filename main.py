import streamlit as st
import yaml

with open("settings/config.yaml", "r") as f:
    config = yaml.safe_load(f)

st.subheader("Road Detection and Segmentation")

# Video file paths
positive_input_path = config["sample_paths"]["positive_input"]
positive_output_path = config["sample_paths"]["positive_output"]
negative_input_path = config["sample_paths"]["negative_input"]
negative_output_path = config["sample_paths"]["negative_output"]

# First row: 2 videos side by side
col1, col2 = st.columns(2)

with col1:
    st.caption("Positive Input")
    st.video(positive_input_path)

with col2:
    st.caption("Positive Output")
    st.video(positive_output_path)

st.subheader("Here's the workflow:")
st.markdown("""
The first layer is an **Autoencoder** used for anomaly detection (one-class classification), trained on positive road scenes to learn the relevant road features.

 Its purpose is to determine whether these features are present in a given image. If the features are detected, the image is passed on to the next layer for detailed pixel-wise segmentation of road and non-road areas.
""")

st.write("The input example above is a positive example, and below is a negative example.")

col3, col4 = st.columns(2)

with col3:
    st.caption("Negative Input")
    st.video(negative_input_path)

with col4:
    st.caption("Negative Output")
    st.video(negative_output_path)

# st.subheader("Step 2: U-Net for Image Segmentation")
st.markdown("""
In the next step, the Modified U-Net receives the image and performs pixel-wise segmentation.

Road pixels are classified as black, while non-road pixels are classified as white.
The segmentation output is then overlayed (translucent) on the original image for better clarity
""")