import cv2
import numpy as np

# Apply monkeypatch to streamlit-webrtc to fix a thread-safety/NoneType issue in python 3.13 or concurrent stop calls
try:
    import streamlit_webrtc.shutdown
    import threading
    import logging

    def _patched_stop(self, timeout: float = 1.0) -> None:
        polling_thread = self._polling_thread
        if polling_thread is not None:
            self._polling_thread_stop_event.set()

            # do not join current thread
            if threading.current_thread() is not polling_thread:
                polling_thread.join(timeout=timeout)
                logger = logging.getLogger("streamlit_webrtc.shutdown")
                if polling_thread.is_alive():
                    logger.warning("ShutdownPolling thread did not exit cleanly")
                else:
                    logger.debug("ShutdownPolling thread stopped cleanly")
            else:
                logging.getLogger("streamlit_webrtc.shutdown").debug("Stop called from polling thread itself, skipping join.")

            self._polling_thread = None

    streamlit_webrtc.shutdown.SessionShutdownObserver.stop = _patched_stop
except ImportError:
    pass

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
    return counts