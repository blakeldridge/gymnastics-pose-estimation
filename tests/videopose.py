import time
import json
import os
import argparse
import numpy as np
import cv2
from models.vitpose import ViTPose

batch_size = 2
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

def run_model(frames, results_path, batch_size=-1):
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
        if len(outputs) == 0:
            continue

        for index, output in enumerate(outputs):
            image_id = batch[index]
            if isinstance(output, list):
                if len(output) == 0:
                    continue
                output = output[0]

            try:
                keypoints = output["keypoints"]
                score = output["score"]
                visibility = np.ones(keypoints.shape[0]).reshape(-1, 1)
                keypoints = np.round(np.hstack([keypoints, visibility]).reshape(-1))
                result = {"image_id": image_id, "category_id": 1, "keypoints": keypoints.tolist(), "score": float(score)}
                results.append(result)
            except Exception as e:
                print("\nBAD OUTPUT DETECTED:")
                print("type:", type(output))
                print("output:", output)

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
                [1, 3, 5, 7, 9, 11, 13, 15],
                [2, 4, 6, 8, 10, 12, 14, 16]
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

def main():
    parser = argparse.ArgumentParser(description="Convert ViTPose JSON results to VideoPose3D NPZ")
    parser.add_argument("--video", required=True, help="Path to video frames directory")
    parser.add_argument("--results-json", required=True, help="Path to ViTPose results JSON")
    parser.add_argument("--output-npz", required=True, help="Path to save VideoPose3D 2D keypoints NPZ")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    results_json = os.path.abspath(args.results_json)
    output_npz = os.path.abspath(args.output_npz)

    frames_path = "results/videopose/frames"

    frames = extract_frames(video_path, frames_path)
    results_json_path = run_model(frames, results_json, batch_size=batch_size)
    vitpose_to_videopose_npz(results_json_path, video_path, frames_path, output_npz, joint_mapping)

if __name__ == "__main__":
    main()