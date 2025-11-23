# Does the inference for a single image through a model of your choosing
# A quick pipeline to get the HPE output of a model

import os
from . import visualisation

DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE = "hr" # "mp"  "hr"  "vp" "mn"

# ---- HRNet ---- #
HR_MODEL_PATH = "../weights/pose_hrnet_w32_256x192.pth"
CONFIG_PATH = "../weights/w32_256x192_adam_lr1e-3.yaml"
# --------------- #

# ---- MediaPipe ---- #
MP_MODEL_PATH = "../weights/pose_landmarker_full.task"
# ------------------- #

IMAGE_PATH = "../data/images/image_6.png"
OUTPUT_PATH = "../results/test1.csv"


def get_path(pth):
    image_path_abs = os.path.join(DIR, pth)
    image_path_abs = os.path.normpath(image_path_abs)
    return image_path_abs

def main():
    image_path = get_path(IMAGE_PATH)
    config_path = get_path(CONFIG_PATH)

    if MODEL_TYPE == "mp":
        from models.mediapipe import MediaPipe

        model_path = get_path(MP_MODEL_PATH)
        model = MediaPipe(model_path)
        output = model(image_path, "device")
        keypoints = output[0][0]["keypoints"]

        visualisation.image_with_joints(image_path, keypoints)

    elif MODEL_TYPE == "hr":
        from models.hrnet import HRNet

        model_path = get_path(HR_MODEL_PATH)
        model = HRNet(model_path, config_path)
        output = model(image_path, image_loc="device")
        keypoints = output[0][0]["keypoints"]
        print(keypoints)

        visualisation.image_with_joints(image_path, keypoints)

    elif MODEL_TYPE == "vp":
        from models.vitpose import ViTPose

        model = ViTPose()
        output = model(image_path, "device")
        keypoints = output[0][0]["keypoints"]

        visualisation.image_with_joints(image_path, keypoints)

    else:
        print("Model not supported!")

if __name__ == "__main__":
    main()