import cv2
from segmentation.yolo_detection import YoloDetector
from selector import CLIPObjectSelector
from utils import draw_bboxes

def webcam_evaluator(clip_model: CLIPObjectSelector):

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        detected_objects, similarities = clip_model.find_objects(frame_rgb)
        print("Similarities:", similarities)

        # Draw bounding boxes and labels on the frame
        frame = draw_bboxes(frame, detected_objects)

        # Display the frame
        cv2.imshow("Webcam Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Load the segmentation model
    segmentation_model = YoloDetector(model_path="./models/yolo12s.pt")

    # Load the CLIP model
    clip_model = CLIPObjectSelector(
        segmentation_model=segmentation_model,
        embedding_model_path="./models/ViT-B-32.pt",
        object_prompt=["A photo of a red water bottle", "A photo of a person wearing glasses", "A photo of a writing pen"],
        device="mps",
        similarity_threshold=0.28,
    )

    # Start the webcam evaluator
    webcam_evaluator(clip_model=clip_model)