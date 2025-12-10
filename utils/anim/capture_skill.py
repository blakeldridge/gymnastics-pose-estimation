import time
import json
import keyboard
import pyautogui
from pathlib import Path

# Video Constants
FRAMES_PER_ANGLE = 45
FRAME_DELAY = 1 / 120 # 120 fps
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

    for angle, (x, y) in angle_buttons.items():
        pyautogui.click(skill_x, skill_y)
        print(f"\nSelecting skill {skill_name}")
        time.sleep(0.5)

        pyautogui.click(*menu_buttons["pause"])
        print(f"Clicked pause button")
        time.sleep(0.2)
        
        pyautogui.click(x, y)
        print("Angle button pressed")
        time.sleep(1.0)

        pyautogui.click(*menu_buttons["pause"])
        print("Clicked unpause")
        print(f"\n-- Capturing angle {angle} --")

        for i in range(FRAMES_PER_ANGLE):
            filename = skill_dir / f"{skill_name}_{angle}_{i:03}.png"
            img = pyautogui.screenshot(region=(
                canvas_region["left"],
                canvas_region["top"],
                canvas_region["width"],
                canvas_region["height"]
            ))
            img.save(filename)
            time.sleep(FRAME_DELAY)

        print(f"Completed {angle}")

        pyautogui.click(*menu_buttons["menu"])
        print("Menu button pressed")
        time.sleep(1.0)
        
    print(f"\n Completed: {skill_name}")
    print(f"Files saved to: {skill_dir}")

if __name__ == "__main__":
    skill = input("Enter skill name (folder prefix): ")
    capture_skill(skill_name=skill)