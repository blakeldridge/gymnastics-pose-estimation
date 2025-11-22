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

class ViTPose:
    def __init__(self, det_model="PekingU/rtdetr_r50vd_coco_o365", vitpose_model="usyd-community/vitpose-base-simple"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_image_processor = AutoProcessor.from_pretrained(det_model)
        self.person_model = RTDetrForObjectDetection.from_pretrained(det_model, device_map=self.device)

        self.image_processor = AutoProcessor.from_pretrained(vitpose_model)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(vitpose_model, device_map=self.device)

    def load_images(self, image_paths, image_loc):
        images = []
        if type(image_paths) == list:
            for path in image_paths:
                if image_loc == "device":
                    images.append(Image.open(path).convert("RGB"))
                else:
                    images.append(Image.open(requests.get(path, stream=True).raw).convert("RGB"))
        else:
            if image_loc == "device":
                images.append(Image.open(image_paths).convert("RGB"))
            else:
                images.append(Image.open(requests.get(image_paths, stream=True).raw).convert("RGB"))

        return images
    
    def detect_persons(self, images):
        inputs = self.person_image_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.person_model(**inputs)

        results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width) for image in images]), threshold=0.3
        )

        boxes = []
        for result in results:
            # Human label refers 0 index in COCO dataset
            person_boxes = result["boxes"][result["labels"] == 0]
            person_boxes = person_boxes.cpu().numpy()

            # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
            
            boxes.append(person_boxes)

        return boxes
    
    def filter_images(self, images, boxes):
        filtered_images = []
        filtered_boxes = []
        for img, box in zip(images, boxes):
            if len(box) > 0:
                filtered_images.append(img)
                filtered_boxes.append(box)

        return filtered_images, filtered_boxes

    def estimate_poses(self, filtered_images, filtered_boxes, boxes):
        inputs = self.image_processor(images=filtered_images, boxes=filtered_boxes, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.pose_model(**inputs)

        pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=filtered_boxes)
        
        keypoints = []
        j = 0
        for box in boxes:
            if len(box) == 0:
                keypoints.append({"keypoints":np.array([[0,0] for _ in range(17)]), "score":0})
            else:
                image = pose_results[j]

                image_kp = []
                for pose in image:
                    image_kp.append({"keypoints":pose["keypoints"].numpy(), "score": np.mean(pose["scores"].numpy())})
                keypoints.append(image_kp)

                j += 1

        return keypoints
    
    def __call__(self, image_paths, image_loc):
        images = self.load_images(image_paths, image_loc)
        boxes = self.detect_persons(images)
        f_images, f_boxes = self.filter_images(images, boxes)

        if len(f_images) == 0:
            return np.array([[[0, 0] for _ in range(17)] for _ in range(len(images))])
        
        keypoints = self.estimate_poses(f_images, f_boxes, boxes)
        return keypoints