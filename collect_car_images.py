
import cv2
import os
import time

CAMERA_INDEX = 1
SAVE_DIR = "car_training_images"
TARGET_IMAGES = 100

os.makedirs(SAVE_DIR, exist_ok=True)

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Cannot open external webcam at index 1.")
    print("Check the camera connection and close other camera apps.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("ParkSync training image collector")
print("C = capture image | Q = quit")
print("Capture the toy car at different positions and angles.")
print("Also capture images with both slots empty.")

count = len([
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

while count < TARGET_IMAGES:
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    display = frame.copy()

    cv2.putText(
        display,
        f"Images saved: {count}/{TARGET_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    cv2.putText(
        display,
        "C = capture | Q = quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("ParkSync - Collect Training Images", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        filename = os.path.join(
            SAVE_DIR,
            f"image_{int(time.time() * 1000)}.jpg"
        )
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved image {count}/{TARGET_IMAGES}: {filename}")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"Collection finished. Images saved: {count}")
print(f"Folder: {os.path.abspath(SAVE_DIR)}")
