import streamlit as st
import cv2
import torch
from detector import ObjectDetector
from utils import draw_boxes, get_counts, inject_custom_css
import tempfile
import os

st.set_page_config(page_title="Video Detection", page_icon="🎬", layout="wide")
inject_custom_css()

st.title("Video Object Detection")
st.markdown("Upload a video to track objects frame-by-frame.")

# --- Sidebar Settings ---
st.sidebar.header("Detection Settings")
model_descriptions = {
    'yolov8n': 'Nano model - Fastest',
    'yolov8s': 'Small model - Balanced',
    'yolov8m': 'Medium model - Accurate',
    'yolov8l': 'Large model - Very Accurate',
    'yolov8x': 'Extra Large model - Best Accuracy'
}
model_options = [f"{model} - {desc}" for model, desc in model_descriptions.items()]
selected_option = st.sidebar.selectbox("Select YOLOv8 Model", model_options, index=0)
selected_model = selected_option.split(' - ')[0]
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.01)

@st.cache_resource
def get_detector(model_name):
    return ObjectDetector(model_name)
detector = get_detector(selected_model)

with st.sidebar.expander("⚙️ Advanced Filters", expanded=False):
    all_class_names = list(detector.names.values())
    options = ["All classes"] + all_class_names
    if 'selections_vid' not in st.session_state:
        st.session_state.selections_vid = ["All classes"]
    selected_options = st.multiselect("What to detect", options, key='multiselect_vid', default=st.session_state.selections_vid)
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
col1, col2 = st.columns([1, 1])
with col1:
    st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
with col2:
    if 'cpu' in str(next(detector.model.model.parameters()).device):
        st.warning("Running on CPU. Video processing might be slow.", icon="⚡")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    st.markdown('<div class="styled-container">', unsafe_allow_html=True)
    metrics_placeholder = st.empty()
    FRAME_WINDOW = st.image([])
    st.markdown('</div>', unsafe_allow_html=True)
    
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
        counts = get_counts(results, detector.names)
        
        # Update metrics live
        if counts:
            with metrics_placeholder.container():
                st.markdown("### Live Analytics")
                items = list(counts.items())
                for i in range(0, len(items), 4):
                    cols = st.columns(4)
                    for j in range(4):
                        if i + j < len(items):
                            cls_name, count = items[i + j]
                            cols[j].metric(label=cls_name.capitalize(), value=count)
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            metrics_placeholder.empty()

        FRAME_WINDOW.image(processed_frame, channels="BGR")
        progress_bar.progress(current_frame / total_frames, text=f"Processing frame {current_frame}/{total_frames}")

    cap.release()
    progress_bar.empty()
    st.success("Video processing complete!")
    os.unlink(video_path)