import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List
from segmentation.detector import Detector


class CLIPObjectSelector:
    """
    Class to select objects from an image based on a prompt using the clip model. Usefull in zero-shot scenarios.
    """
    def __init__(self, 
                object_prompt: List[str],
                embedding_model_path: Path,
                segmentation_model: Detector,
                similarity_threshold: float = 0.5,
                device: str = "mps") -> None:
        
        self.device = device
        self.similarity_threshold = similarity_threshold

        # Initialize the segmentation model
        self.segmentation_model = segmentation_model

        # Initialize the clip model
        self.clip_model, self.preprocess = clip.load(embedding_model_path, device=device)

        # Set the prompt
        self.object_prompt = object_prompt
        self.tokenized_prompts = clip.tokenize(object_prompt).to(device)

    def set_object_prompt(self, object_prompt: List[str]) -> None:
        """
        Set the object prompt for the selector.
        """
        self.object_prompt = object_prompt
    

    def find_objects(self, image: np.ndarray) -> List[Image]:
        """
        Find the object in the image based on the prompt.
        Returns similarities for each cropped region.
        """
        # Perform segmentation
        segmentation_results = self.segmentation_model.predict(image)

        # crop the image based on the segmentation results
        detected_images = self.segmentation_model.crop_detections_squared(segmentation_results)
        cropped_images = [cropped['image'] for cropped in detected_images]

        # images to PIL format and preprocess
        processed_images = torch.stack([self.preprocess(Image.fromarray(cropped)) for cropped in cropped_images]).to(self.device)
        print(f"Processed images shape: {processed_images.shape}")

        # Get the features of all images in batch
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # Get the features of the prompts
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.tokenized_prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity for all crops with temperature scaling
        similarities = (image_features @ text_features.T)  # [N crops, N prompts]

        # Return dict of detected image, similarities and prompt that corresponds to maximum similarity
        objects = []
        for i, cropped in enumerate(detected_images):
            cropped['prompt'] = self.object_prompt[np.argmax(similarities[i,:].cpu().numpy())]
            cropped['similarity'] = np.max(similarities[i,:].cpu().numpy())
            
            if self.similarity_threshold is not None:
                if cropped['similarity'] >= self.similarity_threshold:
                    objects.append(cropped)
            else:
                objects.append(cropped)

        return objects, similarities.cpu().numpy()
    

if __name__ == "__main__":
    import cv2
    from segmentation.yolo_detection import YoloDetector

    # Load the segmentation model
    segmentation_model = YoloDetector(model_path="../models/yolo11m.pt")

    # Load the CLIP model
    clip_model = CLIPObjectSelector(
        object_prompt=["An image of smartphone", "An image of a white person", "A black clicker/remote controller", "An image of a silver frying pan", "A foot wearing flip flops"],
        embedding_model_path='ViT-B/16',
        segmentation_model=segmentation_model,
        device="mps",
        similarity_threshold=None,
    )

    # Read the image
    image = cv2.imread("../resources/pcd0100r.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    images, similarities = clip_model.find_objects(image)

    print("Similarities:", similarities)

    # Show cropped images
    """Draw bounding boxes on the image."""

    # Draw bounding boxes
    for i, detection in enumerate(images):

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
   
    # Display the image with bounding boxes
    cv2.imshow("CLIP Detection", image)
    cv2.waitKey(0)