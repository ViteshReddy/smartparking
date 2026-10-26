
import cv2

print("Checking available cameras...")
print("Look for the window showing your EXTERNAL USB webcam.\n")

for index in range(5):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"Camera index {index}: unavailable")
        camera.release()
        continue

    success, frame = camera.read()

    if success:
        print(f"Camera index {index}: working")
        cv2.imshow(f"Camera Index {index} - Press any key", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Camera index {index}: no image")

    camera.release()

cv2.destroyAllWindows()
print("\nCamera check finished.")
