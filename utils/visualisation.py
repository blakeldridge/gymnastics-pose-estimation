# takes model outputs and image and builds graphs showing image output with joints

import cv2
from PIL import Image, ImageDraw

def image_with_joints(image_path, joints):
    image = cv2.imread(image_path)

    for pt in joints:
        cv2.circle(image, pt.astype(int), radius=2, color=(0,255,0), thickness=-1)

    cv2.imshow("Joint-Image Visualisation", image)
    cv2.waitKey(0)

def draw_comparison(image1_path, image2_path, keypoints1, keypoints2):
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    
    image1_with_keypoints = draw_keypoints(image1, keypoints1)
    image2_with_keypoints = draw_keypoints(image2, keypoints2)

    combined = concat_side_by_side(image1_with_keypoints, image2_with_keypoints)
    return combined

def draw_keypoints(img, keypoints, radius=4, color="red"):
    """Draw circles for keypoints onto a PIL image."""
    draw = ImageDraw.Draw(img)
    for (x, y) in keypoints:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color
        )
    return img


def concat_side_by_side(img1, img2):
    """Concatenate two PIL images horizontally."""
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_img = Image.new("RGB", (w1 + w2, max(h1, h2)))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (w1, 0))
    return new_img

