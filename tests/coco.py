from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import time
from models.vitpose import ViTPose
from models.mediapipe import MediaPipe
from PIL import Image

ANNOTATION_FILE = "/home/blake/Projects/gymnastics-pose-estimation/data/coco_annotations/person_keypoints_val2017.json"
REDUCED_ANNOTATION_FILE = "/home/blake/Projects/gymnastics-pose-estimation/data/coco_annotations/person_keypoints_val2017_reduced.json" # reduced version
DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_NUMBER = 50

ann_file = REDUCED_ANNOTATION_FILE

def coco_evaluation(results_path):
    coco_gt = COCO(ann_file)

    coco_dt = coco_gt.loadRes(results_path)

    img_ids = sorted(coco_gt.getImgIds())

    cocoEval = COCOeval(coco_gt, coco_dt, "keypoints")
    cocoEval.params.imgIds  = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def run_model(model, results_path, batch_size=-1):
    start_time = time.time()

    with open(ann_file, "r") as f:
        data = json.load(f)

    max_images = IMAGE_NUMBER
    image_urls = [img["coco_url"] for img in data["images"][0:max_images]]
    image_ids = [img["id"] for img in data["images"][0:max_images]]

    results = []

    if batch_size == -1:
        batch_size = max_images

    batch_time = time.time()
    for i in range(int(max_images / batch_size)):
        batch_start = batch_size * i
        batch = image_urls[batch_start : batch_start + batch_size]

        outputs = model(batch, image_loc="url")

        for index, output in enumerate(outputs):
            if type(output) == list:
                for pose in output:
                    keypoints = pose["keypoints"]
                    score = pose["score"]
                    visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                    keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                    result = {"image_id": image_ids[batch_start + index], "category_id": 1, "keypoints":keypoints.tolist(), "score": float(score)}
                    results.append(result)
            else:
                keypoints = output["keypoints"]
                score = output["score"]
                visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                result = {"image_id": image_ids[batch_start + index], "category_id": 1, "keypoints":keypoints.tolist(), "score":float(score)}
                results.append(result)

        print(f"Batch {i + 1} Completed : {(time.time() - batch_time):.2f} secs")
        batch_time = time.time()

    results_json = json.dumps(results, indent=4)
    with open(results_path, "w") as f:
        f.write(results_json)
    end_time = time.time()

    print(f"Pose Estiamtion Complete : {(end_time - start_time):.2f} secs")

def test_vitpose():
    results_path = os.path.normpath(os.path.join(DIR, "../results/vitpose_coco_keypoints.json"))

    if not os.path.exists(results_path):
        vitpose = ViTPose()

        run_model(vitpose, results_path, batch_size=5)

    coco_evaluation(results_path)

def reduce_annotations():
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    data["images"] = data["images"][0:IMAGE_NUMBER]

    result_json = json.dumps(data)
    with open(REDUCED_ANNOTATION_FILE, "w") as f:
        f.write(result_json)

if __name__ == "__main__":
    # reduce_annotations()
    test_vitpose()
