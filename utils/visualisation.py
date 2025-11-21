# takes model outputs and image and builds graphs showing image output with joints

import cv2

def image_with_joints(image_path, joints):
    image = cv2.imread(image_path)

    for pt in joints:
        cv2.circle(image, pt.astype(int), radius=2, color=(0,255,0), thickness=-1)

    cv2.imshow("Joint-Image Visualisation", image)
    cv2.waitKey(0)

def image_with_skeleton():
    pass

def skeleton():
    pass

