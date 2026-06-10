import streamlit as st
from utils import inject_custom_css

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# --- Hero Section ---
st.markdown('<div class="hero-title">Real-Time Object Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Powered by YOLOv8 and Streamlit</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --- Feature Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🖼️</div>
            <div class="feature-title">Image Analysis</div>
            <div class="feature-desc">Upload high-resolution images and instantly detect objects with bounding boxes and analytics.</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎬</div>
            <div class="feature-title">Video Tracking</div>
            <div class="feature-desc">Process video files frame-by-frame to track objects over time with adjustable confidence.</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Live Webcam</div>
            <div class="feature-desc">Connect your camera for lightning-fast, real-time object detection directly in your browser.</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.markdown("<hr style='border: 1px solid #e9ecef;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 0.9rem;'>👈 Use the sidebar to select a detection mode and get started.</p>", unsafe_allow_html=True)