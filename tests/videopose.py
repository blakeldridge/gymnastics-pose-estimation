import time
import json
import os
import numpy as np
import cv2
from models.vitpose import ViTPose

video_path = os.path.abspath("data/videos/videopose_test.mp4")
frames_dir = "results/videopose/frames"
results_json = "results/videopose/vitpose_results.json"
output_npz = "results/videopose/data_2d_custom_myvideo.npz"
batch_size = 3
joint_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    frames_dir = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frames_dir.append(frame_filename)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")
    return frames_dir

def run_model(frames, frames_dir, results_path, batch_size=-1):
    start_time = time.time()
    model = ViTPose()
    results = []

    if batch_size == -1:
        batch_size = len(frames)

    batch_time = time.time()
    for i in range(int(np.ceil(len(frames) / batch_size))):
        batch_start = batch_size * i
        batch = frames[batch_start : batch_start + batch_size]

        outputs = model(batch, image_loc="device")

        for index, output in enumerate(outputs):
            image_id = batch[index]
            if isinstance(output, list):
                output = output[0]

            keypoints = output["keypoints"]
            score = output["score"]
            visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
            keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
            result = {"image_id": image_id, "category_id": 1, "keypoints": keypoints.tolist(), "score": float(score)}
            results.append(result)

        print(f"Batch {i + 1} Completed : {(time.time() - batch_time):.2f} secs")
        batch_time = time.time()

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    end_time = time.time()
    print(f"Pose Estimation Complete : {(end_time - start_time):.2f} secs")
    return results_path

def vitpose_to_videopose_npz(results_json_path, video_path, frames_dir, output_path, joint_mapping=None):
    with open(results_json_path, "r") as f:
        results = json.load(f)

    # Frame list
    frames = sorted(os.listdir(frames_dir))
    num_frames = len(frames)

    # Extract video resolution + fps
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Prepare array
    num_joints = len(results[0]["keypoints"]) // 3
    positions_2d = np.zeros((num_frames, num_joints, 2), dtype=np.float32)

    for r in results:
        frame_name = r["image_id"]
        frame_idx = int(os.path.splitext(frame_name)[0].split("_")[1])
        keypoints = np.array(r["keypoints"]).reshape(-1, 3)
        if joint_mapping is not None:
            keypoints = keypoints[joint_mapping]

        positions_2d[frame_idx, :, 0:2] = keypoints[:, 0:2]

    output = {
        "positions_2d": {
            "video_name": {
                "custom": [positions_2d]
            }
        },
        "metadata": {
            "layout_name": "coco",
            "num_joints": num_joints,
            "keypoints_symmetry": (
                [1, 3, 5, 7, 9, 11, 13, 15],   # left side joints
                [2, 4, 6, 8, 10, 12, 14, 16]   # right side joints
            ),
            "video_metadata": {
                "video_name": {
                    "w": width,
                    "h": height,
                    "fps": fps
                }
            }
        }
    }

    np.savez(output_path, **output)
    print(f"Saved VideoPose3D 2D keypoints to {output_path}")

if __name__ == "__main__":
    frames = extract_frames(video_path, frames_dir)
    results_json_path = run_model(frames, frames_dir, results_json, batch_size=batch_size)

    # results_json_path = results_json
    vitpose_to_videopose_npz(results_json_path, video_path, frames_dir, output_npz, joint_mapping)
