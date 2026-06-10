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
    return counts