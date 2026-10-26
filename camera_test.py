
import cv2

# Try camera index 1 first (often the external USB webcam).
camera_index = 1
camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    print("Camera 1 did not open. Trying camera 0...")
    camera_index = 0
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open a camera.")
    print("Check the USB connection and close apps using the webcam.")
    raise SystemExit

print(f"Camera opened successfully: index {camera_index}")
print("Press Q in the camera window to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read a camera frame.")
        break

    cv2.putText(
        frame,
        "Area A - Camera Test",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("ParkSync - Area A Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
