import pyautogui
import time
import json

buttons = ["left", "right", "front", "back", "top"]
menu_buttons = ["pause", "menu"]
coords = {"angle_buttons" : {}, "menu_buttons": {}}

print("\nMove your mouse to EACH angle button when asked.")
print("Press ENTER to capture the coordinate.\n")

time.sleep(2)

for b in buttons:
    input(f"Hover over the '{b}' angle button and press ENTER...")
    x, y = pyautogui.position()
    coords["angle_buttons"][b] = [x, y]
    print(f"Captured {b}: {coords["angle_buttons"][b]}\n")

for b in menu_buttons:
    input(f"Hover over the '{b}' angle button and press ENTER...")
    x, y = pyautogui.position()
    coords["menu_buttons"][b] = [x, y]
    print(f"Captured {b}: {coords["menu_buttons"][b]}\n")

with open("angle_button_coords.json", "w") as f:
    json.dump(coords, f, indent=4)

print("Saved to angle_coords.json")
