# Does the inference for a single image through a model of your choosing
# A quick pipeline to get the HPE output of a model

import os
from . import visualisation

DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE = "vp" # "mp"  "hr"  "vp" "mn"

# ---- HRNet ---- #
MODEL_PATH = "../weights/pose_hrnet_w32_256x192.pth"
CONFIG_PATH = "../weights/w32_256x192_adam_lr1e-3.yaml"
# --------------- #

IMAGE_PATH = "../data/images/image_2.png"
OUTPUT_PATH = "../results/test1.csv"


def get_path(pth):
    image_path_abs = os.path.join(DIR, pth)
    image_path_abs = os.path.normpath(image_path_abs)
    return image_path_abs

def main():
    image_path = get_path(IMAGE_PATH)
    model_path = get_path(MODEL_PATH)
    config_path = get_path(CONFIG_PATH)

    if MODEL_TYPE == "mp":
        pass
    elif MODEL_TYPE == "hr":
        from models.hrnet import run_hrnet
        output = run_hrnet(image_path, model_path, config_path)
        visualisation.image_with_joints(image_path, output[0])
    elif MODEL_TYPE == "vp":
        from models.vitpose import run_vitpose
        output = run_vitpose(image_path)
        visualisation.image_with_joints(image_path, output)
    else:
        print("Model not supported!")

if __name__ == "__main__":
    main()