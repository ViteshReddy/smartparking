
import cv2
import json
import os

CAMERA_INDEX = 1
BOX_FILE = os.path.join(os.path.dirname(__file__), "area_a_boxes.json")

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    CAMERA_INDEX = 0
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open webcam. Close other camera apps and try again.")
    raise SystemExit

print("Camera opened.")
print("Draw a rectangle around Bay 01, then press ENTER or SPACE.")
print("Next, draw a rectangle around Bay 02, then press ENTER or SPACE.")
print("Press C to cancel a rectangle.")

success, frame = camera.read()
camera.release()

if not success:
    print("Could not capture a camera frame.")
    raise SystemExit

frame = cv2.resize(frame, (960, 540))

boxes = []

for bay_name in ["Bay 01", "Bay 02"]:
    print(f"Select the parking space for {bay_name}.")
    box = cv2.selectROI(
        f"Select {bay_name} - drag a box, then press ENTER",
        frame,
        showCrosshair=True,
        fromCenter=False
    )
    cv2.destroyAllWindows()

    x, y, w, h = map(int, box)

    if w == 0 or h == 0:
        print(f"No box selected for {bay_name}. Run the program again.")
        raise SystemExit

    boxes.append({
        "name": bay_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h
    })

preview = frame.copy()

for box in boxes:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        preview, box["name"], (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

cv2.imshow("Area A - Confirm Parking Boxes", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()

with open(BOX_FILE, "w") as file:
    json.dump(boxes, file, indent=4)

print(f"Saved both parking boxes to: {BOX_FILE}")
