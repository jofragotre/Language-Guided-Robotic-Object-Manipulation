import numpy as np
import cv2
from typing import List

def draw_bboxes(image: np.ndarray,
                detections: List[dict]) -> np.ndarray:
    """Draw bounding boxes on the image."""

    # Draw bounding boxes
    for i, detection in enumerate(detections):

        # Get the bounding box coordinates
        box = detection["bbox"]
        x1, y1, x2, y2 = box
        # Convert to integer
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if "prompt" in detection:
            prompt = detection["prompt"]
            cv2.putText(image, prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if "similarity" in detection:
            similarity = detection["similarity"]
            cv2.putText(image, f"{similarity:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image