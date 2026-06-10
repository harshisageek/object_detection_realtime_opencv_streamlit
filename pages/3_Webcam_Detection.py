import streamlit as st
import torch
from detector import ObjectDetector
from utils import draw_boxes, inject_custom_css
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Webcam Detection", page_icon="📸", layout="wide")
inject_custom_css()

st.title("Webcam Live Detection")
st.markdown("Connect your camera for real-time object detection.")

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
    if 'selections_cam' not in st.session_state:
        st.session_state.selections_cam = ["All classes"]
    selected_options = st.multiselect("What to detect", options, key='multiselect_cam', default=st.session_state.selections_cam)
    if selected_options != st.session_state.selections_cam:
        if "All classes" in selected_options and len(selected_options) > 1:
            st.session_state.selections_cam = ["All classes"]
        elif len(selected_options) > 1 and "All classes" in st.session_state:
            st.session_state.selections_cam.remove("All classes")
        elif not selected_options:
            st.session_state.selections_cam = ["All classes"]
        else:
            st.session_state.selections_cam = selected_options
        st.rerun()

selected_class_ids = None
if "All classes" not in st.session_state.selections_cam:
    selected_class_ids = [k for k, v in detector.names.items() if v in st.session_state.selections_cam]

# This class processes the video frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.selected_class_ids = selected_class_ids

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.detector.detect(img, conf_threshold=self.confidence_threshold, classes=self.selected_class_ids)
        processed_img = draw_boxes(img, results, self.detector.names)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- Main Page ---
col1, col2 = st.columns([1, 1])
with col1:
    st.info(f"Model: **{selected_model}** | Device: **{next(detector.model.model.parameters()).device}**")
with col2:
    if 'cpu' in str(next(detector.model.model.parameters()).device):
        st.warning("Running on CPU. Expect lower FPS.", icon="⚡")

st.markdown('<div class="styled-container">', unsafe_allow_html=True)
webrtc_streamer(
    key="webcam",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
st.markdown('</div>', unsafe_allow_html=True)

st.info("Click the 'START' button to begin detection. Grant camera permissions if prompted.")
