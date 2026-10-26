import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("External camera not available at index 1")
    raise SystemExit

while True:
    ok, frame = camera.read()
    if not ok:
        break

    results = model.predict(
        frame,
        classes=[2, 3, 5, 7, 16],
        conf=0.05,
        verbose=False
    )

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{name} {confidence:.0%}",
                (x1, max(25, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    cv2.imshow("Toy Car Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()