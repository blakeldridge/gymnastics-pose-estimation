import os
from models.vitpose import ViTPose
from utils.visualisation import draw_comparison

BLURRY_DIR = "/home/blake/Projects/gymnastics-pose-estimation/data/blurry_images"
DEBLURRED_DIR = "/home/blake/Projects/gymnastics-pose-estimation/data/deblurred_images"
RESULTS_DIR = "/home/blake/Projects/gymnastics-pose-estimation/results/deblur_experiment"

os.makedirs(RESULTS_DIR, exist_ok=True)

def estimate_poses():
    print("Estimating Poses..")
    model = ViTPose()

    blurry_files = sorted(f for f in os.listdir(BLURRY_DIR) if f.endswith(".png"))
    deblurred_files = sorted(f for f in os.listdir(DEBLURRED_DIR) if f.endswith(".png"))

    for i, (blur_name, deblur_name) in enumerate(zip(blurry_files, deblurred_files)):
        blur_path = os.path.join(BLURRY_DIR, blur_name)
        deblur_path = os.path.join(DEBLURRED_DIR, deblur_name)

        outputs = model([blur_path, deblur_path], "device")
        try:
            blur_kp = outputs[0][0]["keypoints"]
        except:
            blur_kp = []
            
        try:
            deblur_kp = outputs[1][0]["keypoints"]
        except:
            deblur_kp = []

        comparison = draw_comparison(blur_path, deblur_path, blur_kp, deblur_kp)

        comparison.save(os.path.join(RESULTS_DIR, f"comparison_{i+1}.png"))
        print(f"Image {i + 1} comparison saved!")
    print("Estimation Finished!")

if __name__ == "__main__":
    estimate_poses()
