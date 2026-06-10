from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path_or_name):
        self.model = YOLO(model_path_or_name)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.names = self.model.names

    def detect(self, frame, conf_threshold=0.40, classes=None):
        # Pass the confidence threshold and class filter to the model
        results = self.model(frame, conf=conf_threshold, classes=classes)
        return results