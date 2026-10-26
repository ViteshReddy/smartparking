
import os
import cv2
from ultralytics import YOLO

IMAGE_DIR = "car_training_images"
OUTPUT_DIR = "car_training_dataset"
MODEL_NAME = "yolo11n.pt"

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

model = YOLO(MODEL_NAME)

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("No images found in car_training_images.")
    raise SystemExit

print(f"Processing {len(image_files)} images...")
print("Generating initial labels automatically.")

for filename in image_files:
    path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    height, width = image.shape[:2]

    results = model.predict(
        image,
        classes=[2],
        conf=0.05,
        verbose=False
    )[0]

    output_image = os.path.join(OUTPUT_DIR, "images", filename)
    output_label = os.path.join(
        OUTPUT_DIR, "labels",
        os.path.splitext(filename)[0] + ".txt"
    )

    cv2.imwrite(output_image, image)

    with open(output_label, "w", encoding="utf-8") as label_file:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # YOLO format: class, center_x, center_y, width, height
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height

            label_file.write(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n"
            )

print("Finished.")
print(f"Images and labels saved in: {OUTPUT_DIR}")
print(
    "IMPORTANT: These are automatic initial labels. "
    "If the model misses the toy car, its label will be empty."
)
