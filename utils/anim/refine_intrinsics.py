import json
import numpy as np
import cv2

# pommel dimensions in pixels (halved)
pommel_length = 377 / 2  # pixels
pommel_width  = 91 / 2
pommel_height = 63 / 2

# pommel dimensions in meters
real_length = 1.7    # meters
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
    "bottom_back_right",
]

# Load annotations
def load_annotations(path):
    with open(path) as f:
        return json.load(f)

def extract_points(ann):
    obj = []
    img = []
    for idx, name in enumerate(POINT_NAMES):
        pt = ann["points"].get(name)
        if pt is not None:
            obj.append(object_points_master[idx])
            img.append(pt)
    return np.array(obj, np.float32), np.array(img, np.float32)

# Calibration function
def refine_intrinsics(annotation_file, width=1920, height=1080):
    anns = load_annotations(annotation_file)

    all_obj_points = []
    all_img_points = []

    for ann in anns:
        obj, img = extract_points(ann)
        if len(obj) >= 4:
            all_obj_points.append(obj)
            all_img_points.append(img)

    print(f"Using {len(all_obj_points)} valid images")

    # initial guess for intrinsics
    fx = fy = 1500
    cx = width / 2
    cy = height / 2

    K_initial = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_initial = np.zeros(5)

    # Lock distortion
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_ZERO_TANGENT_DIST |
        cv2.CALIB_FIX_K1 |
        cv2.CALIB_FIX_K2 |
        cv2.CALIB_FIX_K3 |
        cv2.CALIB_FIX_K4 |
        cv2.CALIB_FIX_K5 |
        cv2.CALIB_FIX_K6
    )

    # Run calibration
    ret, K_refined, dist_refined, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=all_obj_points,
        imagePoints=all_img_points,
        imageSize=(width, height),
        cameraMatrix=K_initial,
        distCoeffs=dist_initial,
        flags=flags
    )

    print("\n=== Refined Intrinsics ===")
    print(K_refined)
    print("\nDistortion (should be zeros):")
    print(dist_refined)

    # Save extrinsics
    results = {
        "K": K_refined.tolist(),
        "dist": dist_refined.tolist(),
        "extrinsics": []
    }

    for ann, rvec, tvec in zip(anns, rvecs, tvecs):
        results["extrinsics"].append({
            "image": ann["image"],
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist()
        })

    with open("refined_camera_data.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved → refined_camera_data.json")

if __name__ == "__main__":
    refine_intrinsics("pommel_annotations.json")
