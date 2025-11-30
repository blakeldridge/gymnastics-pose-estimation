#!/bin/bash

# Activate NAFNet env
source .deblur_env/bin/activate
python -m tests.deblur_images
deactivate

# Activate pose estimation env
source .venv/bin/activate
python -m tests.estimate_poses
deactivate
