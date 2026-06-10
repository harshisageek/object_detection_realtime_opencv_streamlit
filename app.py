import streamlit as st
import os
import base64
from utils import inject_custom_css

st.set_page_config(
    page_title="NEURAL VISION",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Hero Section ---
st.markdown('<div class="hero-container">', unsafe_allow_html=True)

banner_path = os.path.join(os.path.dirname(__file__), 'assets', 'hero_banner.png')
if os.path.exists(banner_path):
    img_b64 = get_base64_of_bin_file(banner_path)
    st.markdown(f'<img src="data:image/png;base64,{img_b64}" style="width: 100%; border-radius: 16px; border: 1px solid rgba(0,229,255,0.4); box-shadow: 0 0 40px rgba(0,229,255,0.2); margin-bottom: 2rem;">', unsafe_allow_html=True)

st.markdown('<div class="hero-title">NEURAL VISION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Real-Time Cyberpunk Object Detection</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --- Feature Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">👁️</div>
            <div class="feature-title">Data Scan</div>
            <div class="feature-desc">Upload static visual data streams. Deep learning models will instantly detect and classify entities.</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎞️</div>
            <div class="feature-title">Video Uplink</div>
            <div class="feature-desc">Process high-framerate video data frame-by-frame with precision confidence thresholds.</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📡</div>
            <div class="feature-title">Live Feed</div>
            <div class="feature-desc">Connect directly to local optics hardware for zero-latency, real-time object tracking.</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 0.9rem; letter-spacing: 1px;'>[ SYSTEM ONLINE - SELECT MODULE IN SIDEBAR ]</p>", unsafe_allow_html=True)