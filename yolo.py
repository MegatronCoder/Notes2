pip install ultralytics

import torch
from IPython.display import display, Image
from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLO model
model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model

def detect_objects(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Perform object detection
    results = model(img)

    # Plot the results
    img_with_boxes = results[0].plot()

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Save the image
    output_path = 'output.jpg'
    cv2.imwrite(output_path, img_rgb)

    # Display the result
    display(Image(filename=output_path))

# Provide the path to the image
image_path = '/content/Screenshot 2024-09-30 225715.png'

# Detect objects in the image
print(f"Detecting objects in {image_path}:")
detect_objects(image_path)
