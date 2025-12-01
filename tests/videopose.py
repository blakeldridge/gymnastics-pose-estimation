import time
import json
import os
import numpy as np
import cv2
from models.vitpose import ViTPose

# --- CONFIGURATION ---
video_path = os.path.abspath("data/videos/videopose_test.mp4")
frames_dir = "results/frames"
results_json = "vitpose_results.json"
output_npz = "data_2d_custom_myvideo.npz"
batch_size = 5  # process all frames at once
joint_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # map to COCO order if needed

# --- 1. Extract frames ---
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
    return frames_dir  # return frame list

# --- 2. Run ViTPose ---
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
            image_id = batch[index]  # frame file name
            if isinstance(output, list):
                for pose in output:
                    keypoints = pose["keypoints"]
                    score = pose["score"]
                    visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                    keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                    result = {"image_id": image_id, "category_id": 1, "keypoints": keypoints.tolist(), "score": float(score)}
                    results.append(result)
            else:
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

# --- 3. Convert to VideoPose3D NPZ ---
def vitpose_to_videopose_npz(results_json_path, frames_dir, output_path, joint_mapping=None):
    with open(results_json_path, "r") as f:
        results = json.load(f)

    frames = sorted(os.listdir(frames_dir))
    num_frames = len(frames)
    num_joints = len(results[0]["keypoints"]) // 3
    positions_2d = np.zeros((num_frames, num_joints, 2), dtype=np.float32)

    for r in results:
        frame_name = r["image_id"]
        frame_idx = int(os.path.splitext(frame_name)[0].split("_")[1])
        keypoints = np.array(r["keypoints"]).reshape(-1, 3)  # x, y, visibility
        if joint_mapping is not None:
            keypoints = keypoints[joint_mapping]

        positions_2d[frame_idx, :, 0:2] = keypoints[:, 0:2]

    np.savez(output_path,
             positions_2d=positions_2d,
             positions_3d=None,
             metadata={"video_name": {"subject": "video_name",
                                      "action": "custom",
                                      "camera": 0}})
    print(f"Saved VideoPose3D 2D keypoints to {output_path}")

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    frames = extract_frames(video_path, frames_dir)
    results_json_path = run_model(frames, frames_dir, results_json, batch_size=batch_size)
    vitpose_to_videopose_npz(results_json_path, frames_dir, output_npz, joint_mapping)
