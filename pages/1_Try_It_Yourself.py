import streamlit as st
import yaml
import os
import time
from backend.inference import make_prediction

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "settings", "config.yaml"))

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

st.subheader("Try it out!")

# Upload a file, inference, play on browser and delete the uploaded video
# st.video("temp/inference_output.mp4")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
# Get the relative path from config
    relative_temp_path = config['inference']['save_path']

    # Convert to absolute path (relative to project root)
    absolute_temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relative_temp_path))

    # Make sure the folder exists
    os.makedirs(absolute_temp_path, exist_ok=True)

    # Create the full path to save the uploaded file
    temp_video_file_path = os.path.join(absolute_temp_path, uploaded_file.name)

    # Save the file
    with open(temp_video_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Video saved to: {temp_video_file_path}")

     # 1) Run inference on the uploaded video
    out_video_path = make_prediction(temp_video_file_path, 1)
    print(f"Output video saved to: {out_video_path}")
    time.sleep(1)  # Wait briefly to ensure file is ready

    st.video(out_video_path)

    output_path = "temp/inference_output.mp4"
    print(f"Size of output video: {os.path.getsize(out_video_path)} bytes")

    # 3) Cleanup: Remove temporary files after displaying the output
    os.remove(temp_video_file_path)
    os.remove(out_video_path)