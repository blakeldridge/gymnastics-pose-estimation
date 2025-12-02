# Arguments
VIDEO_PATH="$1"
OUTPUT_PATH="$2"

if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: $0 <video_path> <output_path>"
    exit 1
fi

OUTPUT_VID="${OUTPUT_PATH}/video.mp4"
OUTPUT_NPZ="${OUTPUT_PATH}/data_2d_custom_myvideo.npz"
OUTPUT_2D="${OUTPUT_PATH}/keypoints_2d.json"
OUTPUT_3D="${OUTPUT_PATH}/keypoints_3d.npz"

# Run videopose file prep
PREP_CMD="python -m tests.videopose --video $VIDEO_PATH --results-json $OUTPUT_2D --output-npz $OUTPUT_NPZ"

# Vitpose Repo path
VIDEOPOSE_PATH="$HOME/Repos/VideoPose3D/"

# run videopose
VIDEOPOSE_CMD="python ${VIDEOPOSE_PATH}run.py -d custom -k myvideo -arc 3,3,3,3,3 --evaluate pretrained_h36m_detectron_coco.bin \
--render --viz-subject video_name --viz-action custom --viz-camera 0 \
--viz-video $VIDEO_PATH --viz-output $OUTPUT_VID --viz-export $OUTPUT_3D"

source .venv/bin/activate
eval "$PREP_CMD"

cp "$OUTPUT_NPZ" "data/data_2d_custom_myvideo.npz" 
echo "NPZ File copied to VideoPose Directory!"

eval "$VIDEOPOSE_CMD"

deactivate