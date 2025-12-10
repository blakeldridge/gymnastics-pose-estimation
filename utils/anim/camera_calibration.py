import json
import numpy as np
import cv2
from pathlib import Path

# pommel dimensions (halved) in pixels
pommel_length = 377 / 2 
pommel_width  = 91 / 2
pommel_height = 63 / 2

# Real pommel dimensions in meters
real_length = 1.7
real_width  = 0.32
real_height = 0.43

# Scale factor to convert pixel object points to Unity-realistic scale
scale_x = real_length / (2 * pommel_length)
scale_y = real_width / (2 * pommel_width)
scale_z = real_height / (2 * pommel_height)

# Object points in 3D (centered at origin)
object_points_master = np.array([
    [-pommel_length*scale_x,  pommel_width*scale_y,   0.0],
    [ pommel_length*scale_x,  pommel_width*scale_y,   0.0],
    [-pommel_length*scale_x, -pommel_width*scale_y,   0.0],
    [ pommel_length*scale_x, -pommel_width*scale_y,   0.0],
    [-pommel_length*scale_x,  pommel_width*scale_y,  -pommel_height*scale_z],
    [ pommel_length*scale_x,  pommel_width*scale_y,  -pommel_height*scale_z],
    [-pommel_length*scale_x, -pommel_width*scale_y,  -pommel_height*scale_z],
    [ pommel_length*scale_x, -pommel_width*scale_y,  -pommel_height*scale_z],
], dtype=np.float32)

POINT_NAMES = [
    "top_front_left",
    "top_front_right",
    "top_back_left",
    "top_back_right",
    "bottom_front_left",
    "bottom_front_right",
    "bottom_back_left",
    "bottom_back_right"
]


def load_annotations(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def extract_correspondences(annotation):
    """
    Returns:
      object_points (Nx3)
      image_points (Nx2)
    Only includes points that were not skipped.
    """
    image_points = []
    obj_points = []

    for idx, name in enumerate(POINT_NAMES):
        pt = annotation["points"][name]
        if pt is not None:
            image_points.append(pt)
            obj_points.append(object_points_master[idx])

    return np.array(obj_points, dtype=np.float32), np.array(image_points, dtype=np.float32)


def main():
    annotations = load_annotations("pommel_annotations.json")

    # Initial guess for intrinsics
    width, height = 1920, 1080
    fx = fy = 1500.0
    cx = width / 2
    cy = height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    print("Initial camera matrix:")
    print(camera_matrix)

    results = []

    for ann in annotations:
        obj_pts, img_pts = extract_correspondences(ann)

        if len(obj_pts) < 4:
            print(f"Not enough points for: {ann['image']}")
            continue

        # Solve camera pose (extrinsics)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            np.zeros(5),
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            print(f"Failed on image: {ann['image']}")
            continue

        # Save results
        results.append({
            "image": ann["image"],
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
        })

        print(f"Calibrated: {ann['image']}  using {len(obj_pts)} points")

    # Write results
    with open("camera_extrinsics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved camera poses to camera_extrinsics.json")


if __name__ == "__main__":
    main()
