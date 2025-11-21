import os
import torch
import requests
from PIL import Image
import numpy as np
from inference_helpers.hrnet import inference, pose_hrnet, transforms
from inference_helpers.hrnet.default import _C as hrnet_cfg
from yacs.config import CfgNode as CN
from torchvision import transforms as tv_transforms

#---------------------------------------------------------#
# - Be aware that this assumes image is cropped to person #
# - should be relatively square image                     #
# - should have a bit of padding around the person        #
#---------------------------------------------------------#

default_weights = "../weights/pose_hrnet_w32_256x192.pth"
default_config = "../weights/w32_256x192_adam_lr1e-3.yaml"

def preprocess_image(cfg, image_path, image_loc):
    # read image
    if image_loc == "device":
        image = Image.open(image_path)
    else:
        image = Image.open(requests.get(image_path, stream=True).raw)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    image = np.array(image)
    h, w = image.shape[:2]
    center = np.array([w/2, h/2], dtype=np.float32)

    # HRNet expects scale in "person size / 200"
    pixel_std = 200
    scale_x = w / pixel_std
    scale_y = h / pixel_std

    scale = np.array([scale_x, scale_y], dtype=np.float32)

    input_w = cfg.MODEL.IMAGE_SIZE[0]
    input_h = cfg.MODEL.IMAGE_SIZE[1]
    input_size = [input_w, input_h]    # [192, 256] typically

    # resize to 256x192
    input_image = transforms.crop(image, center, scale, input_size)

    # normalize
    transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ])
    tensor_image = transform(input_image).unsqueeze(0)
    return tensor_image, center, scale

def run_hrnet(image_path, weights_path=default_weights, config_path=default_config, image_loc="device", device="cpu"):
    # Create config 
    cfg = hrnet_cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    # Load model
    model = pose_hrnet.get_pose_net(cfg, False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    model.to(device)

    # Preprocess image
    image, center, scale = preprocess_image(cfg, image_path, image_loc)
    image.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():  # disable gradients to save memory
        result = model(image).cpu()

    # get joints
    joints, _ = inference.get_final_preds(cfg, result.numpy(), [center], [scale])
    return joints
    