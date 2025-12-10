import json
import numpy as np
import cv2
from pathlib import Path

def load_annotations(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def extract_correspondences(annotation, object_points_master, point_names):
    """
    Returns:
      object_points (Nx3)
      image_points (Nx2)
    Only includes points that were not skipped.
    """
    image_points = []
    obj_points = []

    for idx, name in enumerate(point_names):
        pt = annotation["points"][name]
        if pt is not None:
            image_points.append(pt)
            obj_points.append(object_points_master[idx])

    return np.array(obj_points, dtype=np.float32), np.array(image_points, dtype=np.float32)

def main():
    apparatus = input("Apparatus : ")
    annotations = load_annotations(f"{apparatus}_annotations.json")

    with open("apparatus_data.json", "r") as f:
        data = json.loads(f.read())

    object_points_master = np.array(data["3d_object_points"][apparatus], dtype=np.float32)
    point_names = data["camera_points"][apparatus]

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
        obj_pts, img_pts = extract_correspondences(ann, object_points_master, point_names)

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
