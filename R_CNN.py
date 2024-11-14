git clone https://github.com/facebookresearch/detectron2.git

detectron2
pip install -e .



import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def detect_objects(image_path, predictor, cfg):
    im = cv2.imread(image_path)
    outputs = predictor(im)

    # Get metadata for visualization
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Create a visualizer
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert the image back to BGR for saving with OpenCV
    result_image = out.get_image()[:, :, ::-1]

    # Save the result
    cv2.imwrite("result.jpg", result_image)
    print(f"Result saved as result.jpg")

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return outputs

def main():
    predictor, cfg = setup_model()

    # You can add a loop here to process multiple images
    image_path = "path taka re"  # Replace with your image path
    outputs = detect_objects(image_path, predictor, cfg)

    # Print detected objects and their scores
    instances = outputs["instances"].to("cpu")
    for i in range(len(instances)):
        class_id = instances.pred_classes[i].item()
        class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
        score = instances.scores[i].item()
        print(f"Detected {class_name} with confidence {score:.2f}")

if __name__ == "__main__":
    main()
