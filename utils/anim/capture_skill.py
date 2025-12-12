import time
import json
import keyboard
import pyautogui
import subprocess
import cv2
import os
import threading
from pathlib import Path

# Video Constants
FPS = 30
DURATION = 8
OUTPUT_DIR = "captures"
HOTKEY = "f8"

# Load button positions
with open("angle_button_coords.json", "r") as f:
    button_positions = json.loads(f.read())

angle_buttons = button_positions["angle_buttons"]
menu_buttons = button_positions["menu_buttons"]

# load canvas region
with open("canvas_region_coords.json", "r") as f:
    canvas_region = json.loads(f.read())

region_x = canvas_region["left"]
region_y = canvas_region["top"]
region_width = canvas_region["width"]
region_height = canvas_region["height"]

angles = ["left", "right", "front", "back", "top"]

def record_video(video_path, duration):
    subprocess.run([
        "ffmpeg",
        "-y",
        "-f", "gdigrab",
        "-framerate", str(FPS),
        "-offset_x", str(region_x),
        "-offset_y", str(region_y),
        "-video_size", f"{region_width}x{region_height}",
        "-i", "desktop",
        "-t", str(duration),
        "-c:v", "ffv1",
        video_path
    ])

def preview_with_ffplay(video_path):
    cmd = [
        "ffplay",
        "-vf", "drawtext=text='%{pts\\:hms}':x=10:y=10:fontsize=24",
        "-i", video_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def capture_skill(skill_name):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    skill_dir = Path(OUTPUT_DIR) / skill_name
    skill_dir.mkdir(exist_ok=True)

    print(f"\n--- Skill: {skill_name} ---")
    print(f"Hover over the skill to begin.")
    print(f"Press {HOTKEY} to BEGIN capturing angle frames.\n")

    keyboard.wait(HOTKEY)
    time.sleep(0.35)

    skill_x, skill_y = pyautogui.position()
    time.sleep(0.15)

    pyautogui.click(skill_x, skill_y)
    print(f"\nCapturing {skill_name}")
    time.sleep(0.3)

    threading.Thread(target=record_video, args=(str(skill_dir / f"{skill_name}.avi"), 60), daemon=True).start()

    for angle, (x, y) in angle_buttons.items():
        pyautogui.click(x, y)
        print(f"Camera {angle} button pressed")
        time.sleep(1.0)
        time.sleep(10)

        
    print(f"\n Completed: {skill_name}")
    print(f"Files saved to: {skill_dir}")

import os
import subprocess

def extract_frames(video_path, start, frames, output_folder, angle):
    """
    Extract frames from a video using FFmpeg (frame-perfect, no OpenCV).

    :param video_path: Path to the source video
    :param start: Start time in seconds
    :param frames: Number of frames to extract
    :param output_folder: Folder to save extracted frames
    :param angle: Angle name for naming files
    """
    os.makedirs(output_folder, exist_ok=True)

    # Calculate duration in seconds
    duration = frames / FPS

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", video_path,
        "-ss", str(start),            # start time in seconds
        "-t", str(duration),          # duration to extract
        "-vf", f"fps={FPS}",          # force FPS
        os.path.join(output_folder, f"{angle}_%04d.png")
    ]

    # Run FFmpeg
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    skill = input("Enter skill name (folder prefix): ")
    capture_skill(skill_name=skill)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    skill_dir = Path(OUTPUT_DIR) / skill

    video_path = os.path.join(skill_dir, f"{skill}.avi")

    threading.Thread(target=preview_with_ffplay, args=(str(video_path),), daemon=True).start()

    frame_starts = {}

    for angle in angles:
        frame_starts[angle] = float(input(f"Enter timeframe for camera {angle} : "))

    with open(skill_dir / f"{skill}_starts.json", "w") as f:
         json.dump(frame_starts, f, indent=4)

    with open(skill_dir / f"{skill}_starts.json", "r") as f:
        frame_starts = json.loads(f.read())

    for angle in angles:
        extract_frames(video_path, frame_starts[angle], 30, f"{skill_dir}/frames", angle)

if __name__ == "__main__":
    main()

