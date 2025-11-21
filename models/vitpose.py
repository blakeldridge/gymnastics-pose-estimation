import torch
import os
import requests
import numpy as np
from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

def run_vitpose(image_path, image_loc="device",device="cpu"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if image_loc == "device":
        image = Image.open(image_path)
    else:
        image = Image.open(requests.get(image_path, stream=True).raw)

    h, w = np.array(image).shape[:2]
    image = image.resize((192, 256)).convert("RGB")

    # ------------------------------------------------------------------------
    # Stage 1. Detect humans on the image
    # ------------------------------------------------------------------------

    # You can choose detector by your choice
    person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

    inputs = person_image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]  # take first image results

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # ------------------------------------------------------------------------
    # Stage 2. Detect keypoints for each person found
    # ------------------------------------------------------------------------
    models = ["usyd-community/vitpose-base-simple","usyd-community/vitpose-plus-base","usyd-community/vitpose-plus-large"]
    model_select = models[0]

    image_processor = AutoProcessor.from_pretrained(model_select)
    model = VitPoseForPoseEstimation.from_pretrained(model_select, device_map=device)

    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]  # results for first image
    keypoints = image_pose_result[0]["keypoints"].numpy()

    keypoints_resized = keypoints.copy()
    keypoints_resized[:, 0] *= w / 192
    keypoints_resized[:, 1] *= h / 256
    return keypoints_resized