import streamlit as st
import yaml
import os
from backend.inference import make_prediction

# Load config
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "settings", "config.yaml"))
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

st.subheader("Try it out!")

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# 1st stage: file just uploaded
if uploaded_file is not None and "video_displayed" not in st.session_state:
    relative_temp_path = config['inference']['save_path']
    absolute_temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relative_temp_path))
    os.makedirs(absolute_temp_path, exist_ok=True)

    # Save uploaded video
    temp_video_file_path = os.path.join(absolute_temp_path, uploaded_file.name)
    with open(temp_video_file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Video saved to: {temp_video_file_path}")

    # Run inference and get output path
    out_video_path = make_prediction(temp_video_file_path, 1)

    # Store paths in session state
    st.session_state.temp_video_path = temp_video_file_path
    st.session_state.out_video_path = out_video_path
    st.session_state.video_displayed = True

# 2nd stage: display video and allow cleanup
if "video_displayed" in st.session_state:
    st.video(st.session_state.out_video_path)
    st.success("Inference completed successfully")

    # Optional cleanup button
    if st.button("Clear and upload another"):
        try:
            os.remove(temp_video_file_path)
            os.remove(out_video_path)
        except FileNotFoundError:
            pass
        # Clear session state variables for the next upload
        for key in ["temp_video_path", "out_video_path", "video_displayed"]:
            st.session_state.pop(key, None)
        st.rerun()
