import streamlit as st
import cv2
import torch
from detector import ObjectDetector
from utils import draw_boxes, get_counts, inject_custom_css
from PIL import Image
import numpy as np

st.set_page_config(page_title="Image Detection", page_icon="🖼️", layout="wide")
inject_custom_css()

st.title("Image Object Detection")
st.markdown("Upload an image to detect objects and view analytics.")

# --- Sidebar for Settings ---
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
    if 'selections_img' not in st.session_state:
        st.session_state.selections_img = ["All classes"]
    selected_options = st.multiselect("What to detect", options, key='multiselect_img', default=st.session_state.selections_img)
    if selected_options != st.session_state.selections_img:
        if "All classes" in selected_options and len(selected_options) > 1:
            st.session_state.selections_img = ["All classes"]
        elif len(selected_options) > 1 and "All classes" in st.session_state:
            st.session_state.selections_img.remove("All classes")
        elif not selected_options:
            st.session_state.selections_img = ["All classes"]
        else:
            st.session_state.selections_img = selected_options
        st.rerun()

selected_class_ids = None
if "All classes" not in st.session_state.selections_img:
    selected_class_ids = [k for k, v in detector.names.items() if v in st.session_state.selections_img]

# --- Main Page ---
col1, col2 = st.columns([1, 1])
with col1:
    st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
with col2:
    if 'cpu' in str(next(detector.model.model.parameters()).device):
        st.warning("Running on CPU. Consider GPU for faster processing.", icon="⚡")

image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner("Processing image..."):
        results = detector.detect(frame, conf_threshold=confidence_threshold, classes=selected_class_ids)
        processed_frame = draw_boxes(frame, results, detector.names)
        counts = get_counts(results, detector.names)

    st.markdown('<div class="styled-container">', unsafe_allow_html=True)
    
    if counts:
        st.markdown("### Detection Analytics")
        items = list(counts.items())
        for i in range(0, len(items), 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < len(items):
                    cls_name, count = items[i + j]
                    cols[j].metric(label=cls_name.capitalize(), value=count)
        st.markdown("<hr>", unsafe_allow_html=True)
        
    st.image(processed_frame, channels="BGR", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
