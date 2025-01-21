# YOLO Object Detection Implementation

A step-by-step implementation of the YOLO (You Only Look Once) object detection algorithm using Python, TensorFlow, and OpenCV. This project demonstrates the core components of YOLO v3 through an incremental development approach.

## Project Overview

This repository contains a complete implementation of the YOLO object detection algorithm, built from the ground up. Each file represents a crucial component of the YOLO architecture, allowing for a clear understanding of how object detection systems work.

### Features

- Complete YOLO v3 implementation
- Step-by-step component architecture
- Support for multiple image formats (JPG, JPEG, PNG)
- COCO dataset class detection
- Non-max suppression for overlapping boxes
- Visualization tools for detection results
- Confidence score thresholding
- Anchor box implementation

## Repository Structure

```
├── README.md
├── 0-yolo.py through 7-yolo.py
├── 0-main.py through 7-main.py
├── coco_classes.txt
├── detections/
│   └── [Output images with bounding boxes]
└── yolo_images/
    └── [Test images: dog, eagle, giraffe, horses, person, takagaki]
```

## Implementation Steps

1. **Model Initialization** (0-yolo.py)
   - Loads pre-trained Darknet model
   - Configures model parameters
   - Sets up anchor boxes

2. **Output Processing** (1-yolo.py)
   - Processes raw model outputs
   - Converts network outputs to bounding box predictions
   - Handles feature map processing

3. **Box Filtering** (2-yolo.py)
   - Implements confidence thresholding
   - Filters low-confidence predictions
   - Processes class probabilities

4. **Non-max Suppression** (3-yolo.py)
   - Removes overlapping boxes
   - Implements IoU calculations
   - Handles multiple detections of the same object

5. **Image Loading** (4-yolo.py)
   - Loads images from directory
   - Supports multiple image formats
   - Maintains image path tracking

6. **Image Preprocessing** (5-yolo.py)
   - Resizes images to model input dimensions
   - Normalizes pixel values
   - Prepares batch processing

7. **Box Visualization** (6-yolo.py)
   - Draws bounding boxes
   - Adds class labels and confidence scores
   - Implements image saving functionality

8. **Prediction Pipeline** (7-yolo.py)
   - Combines all components
   - Implements full detection pipeline
   - Handles batch processing of images

## Dependencies

- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy

## Usage

To run the object detection on your images:

```python
from yolo import Yolo

# Initialize the model
yolo = Yolo(model_path, classes_path, class_t, nms_t, anchors)

# Predict on a folder of images
predictions, image_paths = yolo.predict(folder_path)
```

The `predict` method will:
1. Load images from the specified folder
2. Preprocess the images
3. Run detection
4. Display results with bounding boxes
5. Save annotated images when 's' is pressed

## Output

The detection results are saved in the `detections/` folder, with bounding boxes drawn around detected objects, labels showing the class name, and confidence scores for each detection.

## Learning Outcomes

This project demonstrates understanding of:
- Deep learning-based object detection
- Computer vision fundamentals
- Python software architecture
- Image processing techniques
- Neural network output processing
- Performance optimization techniques

## Future Improvements

Potential enhancements for the project:
- Real-time video detection support
- Multiple model backbone support
- Custom training pipeline
- Performance optimizations
- API implementation

## Author

Nathan Rhys - nathan.rhys@atlasschool.com
