import pyautogui
import time
import json

corners = ["top-left", "bottom-right"]
coords = {}

print("\nMove your mouse to EACH angle button when asked.")
print("Press ENTER to capture the coordinate.\n")

time.sleep(2)

for pt in corners:
    input(f"Hover over the '{pt}' and press ENTER...")
    x, y = pyautogui.position()
    coords[pt] = [x, y]
    print(f"Captured {pt}: {coords[pt]}\n")

region = {
    "top": coords["top-left"][1],
    "left": coords["top-left"][0],
    "width": abs(coords["bottom-right"][0] - coords["top-left"][0]),
    "height": abs(coords["bottom-right"][1] - coords["top-left"][1])
}

with open("canvas_region_coords.json", "w") as f:
    json.dump(region, f, indent=4)

print("Saved to canvas_region_coords.json")
