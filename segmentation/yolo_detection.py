from ultralytics import YOLO
from .detector import Detector
from pathlib import Path
from typing import List, Dict
import numpy as np

class YoloDetector(Detector):
    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self.init_model(model_path)

    def init_model(self, model_path: Path) -> None:
        """
        Initialize the YOLO model.
        """
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform inference on the image using the YOLO model.
        """
        results = self.model.predict(image)
        return results


if __name__ == "__main__":

    import cv2 

    # Load the YOLO model
    model = YoloDetector(model_path="../models/yolo11m.pt")

    # Read the image
    image = cv2.imread("../resources/pcd0100r.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(image)

    print(results)
    print(type(results))
    
    # Get the bounding boxes
    boxes = results[0].boxes 
    for box in boxes:

        # Draw the bounding box on the image
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  

    # Display the image with bounding boxes
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()