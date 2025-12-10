import cv2
import json
import os

point_names = []
points = []
skipped = []
current_idx = 0

# Zoom & pan state
zoom_factor = 1.0 
pan_x = 0
pan_y = 0

img = None 

def mouse_callback(event, x, y, flags, param):
    global points, current_idx, zoom_factor, pan_x, pan_y, img

    if event == cv2.EVENT_LBUTTONDOWN and current_idx < len(point_names):
        # Convert display coordinates to original image coordinates
        orig_x = int(x / zoom_factor + pan_x)
        orig_y = int(y / zoom_factor + pan_y)

        # Clamp
        orig_x = max(0, min(orig_x, img.shape[1] - 1))
        orig_y = max(0, min(orig_y, img.shape[0] - 1))

        print(f"Clicked {point_names[current_idx]} at ({orig_x}, {orig_y})")

        points.append([orig_x, orig_y])
        current_idx += 1


def annotate_image(img_path):
    global points, skipped, current_idx
    global zoom_factor, pan_x, pan_y, img

    points = []
    skipped = []
    current_idx = 0

    # Reset zoom/pan each image
    zoom_factor = 1.0
    pan_x = 0
    pan_y = 0

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        return None

    h, w = img.shape[:2]

    cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate", mouse_callback)

    print("\n=======================================")
    print(f"Annotating: {img_path}")
    print("Click the 8 points in this order:")
    for i, name in enumerate(point_names):
        print(f"{i+1}. {name}")
    print("\nControls:")
    print("  Left-click = record point")
    print("  s = skip point")
    print("  u = undo last")
    print("  + / - = zoom in/out")
    print("  Arrow keys = pan view")
    print("  ENTER = save")
    print("  q = quit without saving")
    print("=======================================\n")

    while True:

        # determine cropping region based on pan and zoom 
        view_w = int(w / zoom_factor)
        view_h = int(h / zoom_factor)

        # clamp pan so crop stays in bounds
        pan_x = max(0, min(pan_x, w - view_w))
        pan_y = max(0, min(pan_y, h - view_h))

        # crop
        cropped = img[
            int(pan_y):int(pan_y + view_h),
            int(pan_x):int(pan_x + view_w)
        ]

        # resize to zoom
        display = cv2.resize(
            cropped,
            None,
            fx=zoom_factor,
            fy=zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )

        # draw clicked points
        for i, (px, py) in enumerate(points):
            if pan_x <= px < pan_x + view_w and pan_y <= py < pan_y + view_h:
                dx = int((px - pan_x) * zoom_factor)
                dy = int((py - pan_y) * zoom_factor)

                cv2.circle(display, (dx, dy), 6, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (dx+5, dy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # draw skipped indicators
        for i in range(len(point_names)):
            if i in skipped:
                cv2.putText(display, f"{i+1} (skipped)", (20, 40 + 25*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif i < len(points):
                cv2.putText(display, f"{i+1} (clicked)", (20, 40 + 25*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Annotate", display)
        key = cv2.waitKey(20)

        # Quit
        if key == ord('q'):
            print("Quit without saving.")
            return None

        # Skip
        if key == ord('e') and current_idx < len(point_names):
            print(f"Skipped {point_names[current_idx]}")
            skipped.append(current_idx)
            current_idx += 1

        # Undo
        if key == ord('u'):
            if points:
                removed = points.pop()
                current_idx -= 1
                print(f"Undo point {point_names[current_idx]} at {removed}")
            elif skipped:
                removed = skipped.pop()
                current_idx -= 1
                print(f"Undo skipped {point_names[current_idx]}")
            else:
                print("Nothing to undo.")

        if key == ord('+') or key == ord('='):
            zoom_factor = min(20.0, zoom_factor * 1.2)

        if key == ord('-'):
            zoom_factor = max(1.0, zoom_factor / 1.2)

        step = 50 / zoom_factor
        if key == ord("a"):  # left
            pan_x -= step
        if key == ord("w"):  # up
            pan_y -= step
        if key == ord("d"):  # right
            pan_x += step
        if key == ord("s"):  # down
            pan_y += step

        # Save
        if key == 13:  # ENTER
            if current_idx < len(point_names):
                print("Warning: Not all points completed, saving anyway.")
            break

    cv2.destroyWindow("Annotate")

    # Build output structure
    result = {
        "image": img_path,
        "points": {},
    }
    p_idx = 0
    for i, name in enumerate(point_names):
        if i in skipped:
            result["points"][name] = None
        else:
            result["points"][name] = points[p_idx]
            p_idx += 1

    return result


def main():
    global point_names
    # get apparatus input
    apparatus = input("Apparatus : ")

    # get folder path input
    folder_path = input("Folder path : ")

    # get camera points
    with open("apparatus_data.json", "r") as f:
        data = json.loads(f.read())

    point_names = data["camera_points"][apparatus]
    
    input_image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    output_file = f"{apparatus}_annotations.json"
    all_results = []

    for img_path in input_image_paths:
        result = annotate_image(img_path)
        if result is not None:
            all_results.append(result)
        else:
            print("Annotation stopped early.")
            break

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nSaved annotations to {output_file}")


if __name__ == "__main__":
    main()
