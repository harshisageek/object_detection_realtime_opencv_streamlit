import cv2
import numpy as np

def draw_boxes(frame, results, names):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{names[cls]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame

def get_counts(results, names):
    counts = {name: 0 for name in names.values()}
    for result in results:
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for cls in clss:
            counts[names[cls]] += 1
    # Filter to only return classes with > 0 occurrences
    return {k: v for k, v in counts.items() if v > 0}

import streamlit as st
import os

def inject_custom_css():
    # Use absolute path to ensure Streamlit Cloud finds the file
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load CSS: {e}")