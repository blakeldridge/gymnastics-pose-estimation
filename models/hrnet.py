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
    
    def preprocess_image(self, np_image):
        h, w = np_image.shape[:2]
        center = np.array([w/2, h/2], dtype=np.float32)

        # HRNet expects scale in "person size / 200"
        pixel_std = 200
        scale_x = w / pixel_std
        scale_y = h / pixel_std

        scale = np.array([scale_x, scale_y], dtype=np.float32)

        input_w = self.cfg.MODEL.IMAGE_SIZE[0]
        input_h = self.cfg.MODEL.IMAGE_SIZE[1]
        input_size = [input_w, input_h]    # [192, 256] typically

        # resize to 256x192
        input_image = transforms.crop(np_image, center, scale, input_size)

        # normalize
        transform = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225])
        ])
        tensor_image = transform(input_image).unsqueeze(0)
        return tensor_image, center, scale
    
    def estimate_poses(self, filtered_images, filtered_boxes, all_boxes):
        keypoints = []
        j = 0

        for boxes in all_boxes:
            if len(boxes) == 0:
                keypoints.append([{"keypoints": np.zeros((17,2)), "score": 0}])
                continue

            img_poses = []
            for box in boxes:
                image_w, image_h = filtered_images[j].size
                x1, y1, x2, y2 = box

                # crop image to bbox
                cropped_img = filtered_images[j].crop((x1, y1, x2, y2))
                np_img = np.array(cropped_img)

                # run hrnet on crop
                pre_img, center, scale = self.preprocess_image(np_img)

                self.model.eval()
                with torch.no_grad():
                    result = self.model(pre_img.to(self.device)).cpu()

                # raw HRNet joints (in HRNet input resolution)
                joints, _ = inference.get_final_preds(
                    self.cfg,
                    result.numpy(),
                    [center],
                    [scale]
                )

                joints = joints[0]  # (17,2)

                # map HRNet keypoints -> original image
                mapped = np.zeros_like(joints)
                mapped[:, 0] = joints[:, 0] + x1
                mapped[:, 1] = joints[:, 1] + y1

                img_poses.append({
                    "keypoints": mapped,
                    "score": 1.0    # you can set HRNet heatmap score later
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