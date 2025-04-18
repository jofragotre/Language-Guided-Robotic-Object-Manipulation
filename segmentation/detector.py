from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
import numpy as np
import torch

class Detector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def init_model(self, model_path: Path) -> None:
        """
        Initialize the model. This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform inference on the image. This method should be implemented by subclasses.
        Returns dict with usefull information following yolo standard.
        """
        pass

    @staticmethod
    def crop_detections(predict_results: List[Dict]) -> List[Dict]:
        """
        Create a separate image for each detection.
        """
        
        crop_detections = []
        # Get the bounding boxes
        boxes = predict_results[0].boxes 

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cropped_image = predict_results[0].orig_img[int(y1):int(y2), int(x1):int(x2)]
            crop_detections.append({
                'image': cropped_image,
                'bbox': box.xyxy[0].tolist()
            })
        return crop_detections
    
    @staticmethod
    def crop_detections_squared(predict_results: List[Dict]) -> List[Dict]:
        """
        Create a separate image for each detection and make it square.
        """
        
        crop_detections = []
        # Get the bounding boxes
        boxes = predict_results[0].boxes 

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # Find center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            # Find the size of the square
            size = max(int(x2 - x1), int(y2 - y1))
            # Find the coordinates of the square
            x1 = int(center_x - size / 2)
            y1 = int(center_y - size / 2)
            x2 = int(center_x + size / 2)
            y2 = int(center_y + size / 2)
            # Make sure the coordinates are within the image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(predict_results[0].orig_img.shape[1], x2)
            y2 = min(predict_results[0].orig_img.shape[0], y2)
            # Crop the image
            cropped_image = predict_results[0].orig_img[y1:y2, x1:x2]
            # Append the cropped image and bounding box to the list
            crop_detections.append({
                'image': cropped_image,
                'bbox': [x1, y1, x2, y2]
            })

        return crop_detections
    