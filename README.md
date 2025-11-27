# Gymnastics Pose Estimation : Dissertation

Using state-of-the-art pose estimation techniques applied to gymnastics.

Fujitus is an existing system that uses **multi-camera setups and sensors** to achieve accurate pose estimation.  

My aim is to find an accurate **single-camera pipeline** for gymnastics pose estimation that mimics usual training scenarios.

---

## Baseline Test Results

### ViTPose — COCO Dataset  

| Metric | IoU | Area | MaxDets | Score |
|--------|------|--------|---------|-------|
| **Average Precision (AP)** | 0.50:0.95 | all | 20 | **0.768** |
| **Average Precision (AP)** | 0.50 | all | 20 | **0.918** |
| **Average Precision (AP)** | 0.75 | all | 20 | **0.864** |
| **Average Precision (AP)** | 0.50:0.95 | medium | 20 | **0.819** |
| **Average Precision (AP)** | 0.50:0.95 | large | 20 | **0.770** |
| **Average Recall (AR)**    | 0.50:0.95 | all | 20 | **0.835** |
| **Average Recall (AR)**    | 0.50 | all | 20 | **0.978** |
| **Average Recall (AR)**    | 0.75 | all | 20 | **0.913** |
| **Average Recall (AR)**    | 0.50:0.95 | medium | 20 | **0.859** |
| **Average Recall (AR)**    | 0.50:0.95 | large | 20 | **0.794** |

> Model run on 50 images

### HRNet - COCO Dataset

| Metric | IoU | Area | MaxDets | Score |
|--------|-----|------|---------|-------|
| **Average Precision (AP)** | 0.50:0.95 | all | 20 | **0.752** |
| **Average Precision (AP)** | 0.50 | all | 20 | **0.922** |
| **Average Precision (AP)** | 0.75 | all | 20 | **0.889** |
| **Average Precision (AP)** | 0.50:0.95 | medium | 20 | **0.774** |
| **Average Precision (AP)** | 0.50:0.95 | large | 20 | **0.774** |
| **Average Recall (AR)**    | 0.50:0.95 | all | 20 | **0.789** |
| **Average Recall (AR)**    | 0.50 | all | 20 | **0.957** |
| **Average Recall (AR)**    | 0.75 | all | 20 | **0.913** |
| **Average Recall (AR)**    | 0.50:0.95 | medium | 20 | **0.793** |
| **Average Recall (AR)**    | 0.50:0.95 | large | 20 | **0.794** |

> Model run on 50 images

### Mediapipe — COCO Dataset

| Metric | IoU | Area | MaxDets | Score |
|--------|-----|------|---------|-------|
| **Average Precision (AP)** | 0.50:0.95 | all | 20 | **0.095** |
| **Average Precision (AP)** | 0.50 | all | 20 | **0.152** |
| **Average Precision (AP)** | 0.75 | all | 20 | **0.099** |
| **Average Precision (AP)** | 0.50:0.95 | medium | 20 | **0.062** |
| **Average Precision (AP)** | 0.50:0.95 | large | 20 | **0.134** |
| **Average Recall (AR)**    | 0.50:0.95 | all | 20 | **0.102** |
| **Average Recall (AR)**    | 0.50 | all | 20 | **0.152** |
| **Average Recall (AR)**    | 0.75 | all | 20 | **0.109** |
| **Average Recall (AR)**    | 0.50:0.95 | medium | 20 | **0.062** |
| **Average Recall (AR)**    | 0.50:0.95 | large | 20 | **0.171** |

> Model run on 50 images

There is a clear lack of accuracy with this model, reasoning :
- lightweight model
- not detecting a lot of poses
- might be worth using a better bounding box detector first

---

## Input Data, Annotations, and Weights

### COCO  
- Using annotations from the 2017 validation images: [COCO Download](https://cocodataset.org/#download)  
- Stored at: `data/coco_annotations/person_keypoints_val2017.json`  

### HRNet  
- Using the **HRNet-W32 256x192 pre-trained weights**: [Google Drive](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)  
- Stored at: `weights/pose_hrnet_w32_256x192.pth`  
- Config file: [HRNet COCO Config](https://github.com/HRNet/HRNet-Human-Pose-Estimation/tree/master/experiments/coco/hrnet)  
- Stored at: `weights/w32_256x192_adam_lr1e-3.yaml`  

### Mediapipe

- Using the **Full pose landmarker pre-trained weights** : [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models)
- Stored at : `weights/pose_landmarker_full.task`

---

## Libraries Used

- PyTorch  
- NumPy  
- Pillow  
- YACS  
- Requests  
- pycocotools  
- OpenCV (`opencv-python`)  
- mediapipe

---
