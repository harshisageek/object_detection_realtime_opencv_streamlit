import streamlit as st

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("YOLO Object Detection")

# --- Page Layout ---
col1, col2 = st.columns(2)

with col1:
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb212NTd6cDU2bXV5NWhjenk2Z3VjN2o1Z3Z3bWQxbmN2b3hzaXFmMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPnAiaMCws8nOsE/giphy.gif")

with col2:
    st.markdown("""
    ### This app showcases the use of YOLOv8 for object detection.

    You can analyze images, videos, or a live webcam feed.

    ---

    #### **How to Use:**
    1.  **Select a mode** from the sidebar.
    2.  **Choose a YOLOv8 model** (from the speedy 'n' to the powerful 'x').
    3.  **Adjust the confidence threshold** to filter out uncertain detections.
    4.  **Pick specific object classes** to focus your analysis.

    ---
    Have fun
    """)