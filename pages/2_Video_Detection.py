import streamlit as st
import cv2
import torch
from detector import ObjectDetector
from utils import draw_boxes
import tempfile
import os

st.set_page_config(page_title="Video Detection", page_icon="�", layout="wide")
st.title("Video Object Detection")

# --- Sidebar Settings ---
st.sidebar.header("Detection Settings")
model_descriptions = {
    'yolov8n': 'Nano model - Fastest but less accurate',
    'yolov8s': 'Small model - Good balance of speed and accuracy',
    'yolov8m': 'Medium model - Higher accuracy, moderate speed',
    'yolov8l': 'Large model - Very high accuracy, slower speed',
    'yolov8x': 'Extra Large model - Highest accuracy, slowest speed'
}
model_options = [f"{model} - {desc}" for model, desc in model_descriptions.items()]
selected_option = st.sidebar.selectbox("Select YOLOv8 Model", model_options, index=0)
selected_model = selected_option.split(' - ')[0]
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.01)

@st.cache_resource
def get_detector(model_name):
    return ObjectDetector(model_name)
detector = get_detector(selected_model)

all_class_names = list(detector.names.values())
options = ["All classes"] + all_class_names
if 'selections_vid' not in st.session_state:
    st.session_state.selections_vid = ["All classes"]
selected_options = st.sidebar.multiselect("What to detect", options, key='multiselect_vid', default=st.session_state.selections_vid)
if selected_options != st.session_state.selections_vid:
    if "All classes" in selected_options and len(selected_options) > 1:
        st.session_state.selections_vid = ["All classes"]
    elif len(selected_options) > 1 and "All classes" in st.session_state:
        st.session_state.selections_vid.remove("All classes")
    elif not selected_options:
        st.session_state.selections_vid = ["All classes"]
    else:
        st.session_state.selections_vid = selected_options
    st.rerun()
selected_class_ids = None
if "All classes" not in st.session_state.selections_vid:
    selected_class_ids = [k for k, v in detector.names.items() if v in st.session_state.selections_vid]

# --- Main Page ---
st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    FRAME_WINDOW = st.image([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Processing...")

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        results = detector.detect(frame, conf_threshold=confidence_threshold, classes=selected_class_ids)
        processed_frame = draw_boxes(frame, results, detector.names)

        FRAME_WINDOW.image(processed_frame, channels="BGR")

        progress_bar.progress(current_frame / total_frames, text=f"Processing frame {current_frame}/{total_frames}")

    cap.release()
    progress_bar.empty()
    st.success("Video processing complete!")
    os.unlink(video_path)
else:
    st.info("Upload a video to get started.")

if 'cpu' in str(next(detector.model.model.parameters()).device):
    st.warning("Note: The model is running on CPU, so video processing might be slow or laggy. For real-time performance, a GPU is recommended.", icon="⚙️")