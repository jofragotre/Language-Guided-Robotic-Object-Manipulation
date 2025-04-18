from ultralytics import SAM
from .detector import Detector
from pathlib import Path
from typing import List, Dict
import numpy as np

class SAMDetector(Detector):
    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        self.init_model(model_path)

    def init_model(self, model_path: Path) -> None:
        """
        Initialize the SAM model.
        """
        self.model = SAM(model_path)

    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform inference on the image using the SAM model.
        """
        results = self.model.predict(image)
        return results
    

if __name__ == "__main__":

    import cv2 

    # Load the SAM model
    model = SAMDetector(model_path="../models/sam2.1_s.pt")

    # Read the image
    image = cv2.imread("../resources/pcd1002r.png")
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
    cv2.imshow("SAM Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()