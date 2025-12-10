import cv2
import numpy as np
import json
from models.vitpose import ViTPose
from utils.visualisation import image_with_joints

pose_model = ViTPose()

camera_names = ["front", "back", "left", "right", "top"]

def triangulate_joint(joint_idx, frame_idx):
    points_2d = []
    proj_matrices = []

    for cam_name in camera_names:
        kp = all_2d_keypoints[cam_name][frame_idx, joint_idx]
        # Convert rvec/tvec to rotation matrix
        R, _ = cv2.Rodrigues(rvecs[cam_name])
        t = tvecs[cam_name].reshape(3,1)
        P = camera_matrices[cam_name] @ np.hstack((R, t))
        proj_matrices.append(P)
        points_2d.append(kp.reshape(2,1))

    # OpenCV triangulation
    # Linear least square
    A = []
    for pt, P in zip(points_2d, proj_matrices):
        x, y = pt.flatten()
        A.append(x*P[2,:] - P[0,:])
        A.append(y*P[2,:] - P[1,:])
    A = np.stack(A)
    _, _, V = np.linalg.svd(A)
    X = V[-1]
    X /= X[3]  # homogeneous to 3D
    return X[:3]

all_2d_keypoints = {}
camera_matrices = {}
dist_coeffs = {}
rvecs = {}
tvecs = {}

for cam in camera_names:
    frames = [f"data/mocap/magyar_{cam}_000.png"]
    cam_keypoints = []
    for frame_path in frames:
        outputs = pose_model(frame_path, image_loc="device")
        cam_keypoints.append(outputs[0][0]["keypoints"])
    all_2d_keypoints[cam] = np.array(cam_keypoints)
    print(f"Camera {cam} Poses estimated!")

# Load extrinsics & intrinsics
with open("data/mocap/refined_camera_data.json") as f:
    calib = json.load(f)

for cam in calib["extrinsics"]:
    name = cam["image"].split("_")[1]
    camera_matrices[name] = np.array(calib["K"])
    dist_coeffs[name] = np.array(calib["dist"])
    rvecs[name] = np.array(cam["rvec"])
    tvecs[name] = np.array(cam["tvec"])

# Triangulate all joints for all frames
num_joints = all_2d_keypoints["front"].shape[1]
num_frames = all_2d_keypoints["front"].shape[0]

all_3d_keypoints = np.zeros((num_frames, num_joints, 3))
for f in range(num_frames):
    for j in range(num_joints):
        all_3d_keypoints[f,j] = triangulate_joint(j, f)

all_2d_proj = {cam: [] for cam in camera_names}
for cam_name in camera_names:
    K = camera_matrices[cam_name]
    R, _ = cv2.Rodrigues(rvecs[cam_name])
    t = tvecs[cam_name].reshape(3,1)
    for f in range(num_frames):
        pts_3d = all_3d_keypoints[f]
        pts_2d, _ = cv2.projectPoints(pts_3d, rvecs[cam_name], tvecs[cam_name], K, dist_coeffs[cam_name])
        all_2d_proj[cam_name].append(pts_2d.reshape(-1,2))

print("\n -- 3D keypoints -- ")
print(all_3d_keypoints)
print("\n\n -- 2D keypoints -- ")
print(all_2d_proj)

for cam in camera_names:
    img_path = f"data/mocap/magyar_{cam}_000.png"
    keypoints = all_2d_keypoints[cam][0]

    image_with_joints(img_path, keypoints)