```markdown
# Language-Guided-Zero-Shot-Object-Detection

This project implements a simple pipeline for zero-shot object detection using language guidance. It combines a VLM model in this case OpenAI's CLIP for image-text similarity and YOLO for object detection, enabling the detection and classification of objects in images based on natural language prompts.

## TODO:
```markdown
- Accelerate SAM with lightweight Ultralytics models for live use.
- Implement Non-Maximum Suppression (NMS) for handling multiple detections of the same object.
- Integrate a detection model with support for more classes (e.g., SAM) since YOLO's pretrained model supports only 80 classes, or alternatively, train a custom detection model.
```

## Features
- **Zero-Shot Object Detection**: Detect objects in images without requiring task-specific training.
- **Language-Guided Detection**: Use natural language prompts to identify and classify objects.
- **YOLO Integration**: Leverages YOLO for object detection and segmentation.
- **CLIP Integration**: Uses CLIP to match detected objects with text prompts based on similarity.
- **Customizable Prompts**: Easily define prompts to guide object detection.

## Project Structure
- **`segmentation/`**: Contains YOLO-based segmentation and detection logic.
  - `yolo_detection.py`: Implements YOLO-based object detection.
  - `sam_detection.py`: Implements SAM-based object detection.
- **`selector.py`**: Implements the CLIP-based object selection pipeline, matching detected objects with text prompts.
- **`models/`**: Stores pre-trained YOLO and CLIP model files.
- **`resources/`**: Contains sample images for testing the pipeline.
- **`README.md`**: Documentation for the project.

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO and SAM
- CLIP (via OpenAI's `clip` library)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Language-Guided-Zero-Shot-Object-Detection.git
   cd Language-Guided-Zero-Shot-Object-Detection

   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download the required model files:
   - Place the YOLO model (e.g., `yolo11m.pt`) in the models directory.
   - Ensure CLIP downloads its models automatically or place them in the appropriate cache directory.

## Usage
1. Run the webcam demo script:
    ```bash
    python segmentation/webcam_demo.py
    ```

2. Modify the prompts in `webcam_demo.py` to customize the object detection behavior:
    ```python
    self.object_prompt = [
         "A black smartphone",
         "A person wearing a red shirt",
         "A remote controller",
         "A frying pan"
    ]
    ```

3. Use the webcam feed to detect objects in real-time based on the defined prompts.

4. View the results, including bounding boxes and matched prompts, displayed directly on the webcam feed.

## Example
Given an input image and the prompt `"A black smartphone"`, the pipeline will:
1. Use YOLO to detect objects in the image.
2. Crop the detected objects.
3. Use CLIP to calculate the similarity between the cropped objects and the prompt.
4. Highlight the object that best matches the prompt.

## Troubleshooting
- **File Not Found Errors**: Ensure all relative paths are correct and run the scripts from the project root directory.
- **Close Similarity Scores**: Verify the preprocessing pipeline and ensure the prompts are specific and descriptive.
- **Model Download Issues**: Check your internet connection and SSL settings if models fail to download.