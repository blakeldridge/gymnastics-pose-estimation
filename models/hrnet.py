import os
import torch
import requests
from PIL import Image
import numpy as np
from inference_helpers.hrnet import inference, pose_hrnet, transforms
from inference_helpers.hrnet.default import _C as hrnet_cfg
from yacs.config import CfgNode as CN
from torchvision import transforms as tv_transforms

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
)

#---------------------------------------------------------#
# - Be aware that this assumes image is cropped to person #
# - should be relatively square image                     #
# - should have a bit of padding around the person        #
#---------------------------------------------------------#

preprocess = tv_transforms.Compose([
    tv_transforms.ToTensor(),  # converts HxWxC to CxHxW and [0,255] -> [0,1]
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

default_weights = "../weights/pose_hrnet_w32_256x192.pth"
default_config = "../weights/w32_256x192_adam_lr1e-3.yaml"

class HRNet:
    def __init__(self, weights_path, config_path, det_model="PekingU/rtdetr_r50vd_coco_o365"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cfg = hrnet_cfg.clone()
        self.cfg.merge_from_file(config_path)
        self.cfg.freeze()

        self.person_image_processor = AutoProcessor.from_pretrained(det_model)
        self.person_model = RTDetrForObjectDetection.from_pretrained(det_model, device_map=self.device)

        # Load model
        self.model = pose_hrnet.get_pose_net(self.cfg, False)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        self.model.to(self.device)
        self.model.eval()

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
    
    def box_to_center_scale(self, box, aspect_ratio=0.75, pixel_std=200):
        """
        Convert bounding box to center and scale.
        aspect_ratio: width / height ratio for the model input (192/256 = 0.75)
        """
        x1, y1, x2, y2 = box
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        
        # Add padding factor (typically 1.25 for better coverage)
        if self.cfg.MODEL.get('EXTRA', {}).get('SCALE_FACTOR', None):
            scale = scale * 1.25

        return center, scale

    def estimate_poses(self, filtered_images, filtered_boxes, all_boxes):
        keypoints = []
        j = 0

        for boxes in all_boxes:
            if len(boxes) == 0:
                keypoints.append([{"keypoints": np.zeros((17, 3)), "score": 0}])
                continue

            img_poses = []
            batch_centers = []
            batch_scales = []
            batch_images = []
            
            for box in boxes:
                image = filtered_images[j]
                
                # Use proper box_to_center_scale with aspect ratio
                center, scale = self.box_to_center_scale(
                    box, 
                    aspect_ratio=float(self.cfg.MODEL.IMAGE_SIZE[0]) / self.cfg.MODEL.IMAGE_SIZE[1],
                    pixel_std=200
                )
                
                batch_centers.append(center)
                batch_scales.append(scale)
                
                # Crop and transform image
                input_image = transforms.crop(
                    np.array(image), 
                    center, 
                    scale, 
                    self.cfg.MODEL.IMAGE_SIZE  # [width, height] = [192, 256]
                )
                
                # Preprocess
                pre_img = preprocess(input_image)
                batch_images.append(pre_img)

            # Batch inference
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                result = self.model(batch_tensor)
            
            # Move to CPU and convert to numpy
            heatmaps = result.cpu().numpy()
            
            # Get predictions with scores
            preds, maxvals = inference.get_final_preds(
                self.cfg,
                heatmaps,
                batch_centers,
                batch_scales
            )

            # Process each person's pose
            for idx in range(len(boxes)):
                # Combine keypoints with confidence scores
                kps = preds[idx]

                # Calculate overall score as mean of joint confidences
                overall_score = float(np.mean(maxvals[idx]))
                
                img_poses.append({
                    "keypoints": kps,
                    "score": overall_score
                })

            keypoints.append(img_poses)
            j += 1

        return keypoints

    def __call__(self, image_paths, image_loc):
        images = self.load_images(image_paths, image_loc)
        boxes = self.detect_persons(images)
        f_images, f_boxes = self.filter_images(images, boxes)
        keypoints = self.estimate_poses(f_images, f_boxes, boxes)
        return keypoints