import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import requests
import numpy as np

MP_TO_COCO_IDX = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

class MediaPipe:
    def __init__(self, model_path):
        BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=5
        )

    def load_mp_image(self, path, image_loc):
        if image_loc == "device":
            image = Image.open(path).convert("RGB")
        else:
            image = Image.open(requests.get(path, stream=True).raw).convert("RGB")

        image_arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
        return mp_image, image_arr

    def load_images(self, image_paths, image_loc):
        images = []
        image_arrays = []
        if type(image_paths) == list:
            for path in image_paths:
                mp_img, np_img = self.load_mp_image(path, image_loc)
                images.append(mp_img)
                image_arrays.append(np_img)
        else:
            mp_img, np_img = self.load_mp_image(image_paths, image_loc)
            images.append(mp_img)
            image_arrays.append(np_img)

        return images, image_arrays

    def estimate_pose(self, images):
        poses = []
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            for img in images:
                img_poses = []
                pose_landmarker_result = landmarker.detect(img)

                if not pose_landmarker_result.pose_landmarks:
                    poses.append([{"keypoints":np.array([[0, 0] for _ in range(33)]), "score":0}])
                    continue

                for person in pose_landmarker_result.pose_landmarks:
                    keypoints_list = []
                    confidence_list = []

                    # pose_landmarks is a list of normalized landmarks
                    for landmark in person:
                        keypoints_list.append([landmark.x, landmark.y])
                        confidence_list.append(landmark.visibility)

                    keypoints = np.array(keypoints_list)
                    score = float(np.mean(confidence_list))

                    img_poses.append({"keypoints": keypoints, "score": score})
                poses.append(img_poses)

        return poses

    def convert_coco_format(self, mp_results):
        coco_results = []
        for image in mp_results:
            image_results = []
            for person in image:
                mp_keypoints = person["keypoints"] 
                score = person["score"]

                coco_keypoints = mp_keypoints[MP_TO_COCO_IDX]

                image_results.append({
                    "keypoints": coco_keypoints,
                    "score": score
                })
            coco_results.append(image_results)

        return coco_results
    
    def keypoints_to_pixels(self, keypoints, images):
        for i, image in enumerate(images):
            height, width = image.shape[:2]
            for pose in keypoints[i]:
                keypoints_norm = pose["keypoints"]
                keypoints_px = np.zeros_like(keypoints_norm)

                keypoints_px[:, 0] = keypoints_norm[:, 0] * width 
                keypoints_px[:, 1] = keypoints_norm[:, 1] * height

                pose["keypoints"] = keypoints_px

        return keypoints

    def __call__(self, image_paths, image_loc):
        images, image_arrays = self.load_images(image_paths, image_loc)
        mp_keypoints = self.estimate_pose(images)
        keypoints = self.convert_coco_format(mp_keypoints)
        keypoints = self.keypoints_to_pixels(keypoints, image_arrays)
        return keypoints