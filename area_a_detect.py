import cv2
import numpy as np
import os
from collections import deque

# ParkSync - Area A
CAMERA_INDEX = 1
REFERENCE_FILE = "area_a_empty_reference.npz"

DIFF_THRESHOLD = 25
OCCUPIED_THRESHOLD = 0.16
EMPTY_THRESHOLD = 0.09
VOTE_FRAMES = 7


def find_slots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not 12000 < area < 450000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.025 * perimeter, True
        )

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if w < 100 or h < 100:
            continue

        ratio = w / float(h)
        if not 0.35 < ratio < 2.8:
            continue

        duplicate = False
        for ox, oy, ow, oh in candidates:
            if (
                abs(x - ox) < 18
                and abs(y - oy) < 18
                and abs(w - ow) < 25
                and abs(h - oh) < 25
            ):
                duplicate = True
                break

        if not duplicate:
            candidates.append((x, y, w, h))

    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    slots = candidates[:2]
    slots.sort(key=lambda b: b[0])
    return slots


def get_slot_image(frame, box):
    x, y, w, h = box

    # Ignore the outline itself; compare the inner parking-space area.
    px = max(5, int(w * 0.09))
    py = max(5, int(h * 0.09))

    crop = frame[y + py:y + h - py, x + px:x + w - px]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (160, 160))


def difference_score(current, reference):
    if current is None or reference is None:
        return 0.0

    current = cv2.resize(current, (160, 160))
    reference = cv2.resize(reference, (160, 160))

    # Both inputs are BGR images. Convert both consistently to LAB.
    current_lab = cv2.cvtColor(
        current.astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    reference_lab = cv2.cvtColor(
        reference.astype(np.uint8), cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    # Compensate for small overall brightness changes.
    shift = np.median(
        current_lab[:, :, 0] - reference_lab[:, :, 0]
    )
    current_lab[:, :, 0] -= shift

    diff = np.abs(current_lab - reference_lab).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    changed = (
        (diff[:, :, 0] > DIFF_THRESHOLD)
        | (diff[:, :, 1] > DIFF_THRESHOLD)
        | (diff[:, :, 2] > DIFF_THRESHOLD)
    )

    mask = changed.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    return float(np.count_nonzero(mask)) / mask.size


camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Cannot open camera index 1. Check the external USB webcam.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

references = [None, None]
occupied = [False, False]
scores = [0.0, 0.0]
votes = [
    deque(maxlen=VOTE_FRAMES),
    deque(maxlen=VOTE_FRAMES)
]

# Do not reuse old references from the previous buggy version.
if os.path.exists(REFERENCE_FILE):
    try:
        os.remove(REFERENCE_FILE)
        print("Removed old reference file; capture new empty references.")
    except OSError:
        print("Close any program using the reference file and restart.")

print("ParkSync Area A camera started.")
print("Keep the camera fixed.")
print("E = capture empty slots | R = reset | Q = quit")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    display = frame.copy()
    slots = find_slots(frame)

    if len(slots) != 2:
        cv2.putText(
            display,
            f"Parking outlines found: {len(slots)}/2",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2
        )
    else:
        for i, box in enumerate(slots):
            x, y, w, h = box
            crop = get_slot_image(frame, box)

            if references[i] is not None and crop is not None:
                scores[i] = difference_score(crop, references[i])

                threshold = (
                    EMPTY_THRESHOLD if occupied[i]
                    else OCCUPIED_THRESHOLD
                )
                votes[i].append(scores[i] > threshold)

                if len(votes[i]) == VOTE_FRAMES:
                    vote_count = sum(votes[i])

                    if vote_count >= 5:
                        occupied[i] = True
                    elif vote_count <= 2:
                        occupied[i] = False

            color = (0, 0, 255) if occupied[i] else (0, 220, 0)
            status = "OCCUPIED" if occupied[i] else "AVAILABLE"

            cv2.rectangle(
                display, (x, y), (x + w, y + h), color, 3
            )
            cv2.putText(
                display,
                f"SLOT {i + 1}: {status}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            if references[i] is not None:
                cv2.putText(
                    display,
                    f"Change: {scores[i] * 100:.1f}%",
                    (x, min(frame.shape[0] - 10, y + h + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

        cv2.putText(
            display,
            f"AREA A | Occupied: {sum(occupied)}/2 | Outlines: 2/2",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if any(ref is None for ref in references):
            cv2.putText(
                display,
                "Empty both slots, then press E",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2
            )

    cv2.putText(
        display,
        "E: capture empty slots   R: reset   Q: quit",
        (20, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("ParkSync - Area A Occupancy", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("e") and len(slots) == 2:
        crops = [get_slot_image(frame, box) for box in slots]

        if all(crop is not None for crop in crops):
            references = [crop.copy() for crop in crops]
            occupied = [False, False]
            scores = [0.0, 0.0]
            votes = [
                deque(maxlen=VOTE_FRAMES),
                deque(maxlen=VOTE_FRAMES)
            ]

            np.savez(
                REFERENCE_FILE,
                slot1=references[0],
                slot2=references[1]
            )
            print("Captured fresh empty references for both slots.")

    elif key == ord("r"):
        references = [None, None]
        occupied = [False, False]
        scores = [0.0, 0.0]
        votes = [
            deque(maxlen=VOTE_FRAMES),
            deque(maxlen=VOTE_FRAMES)
        ]

        if os.path.exists(REFERENCE_FILE):
            os.remove(REFERENCE_FILE)

        print("Reset complete. Empty both slots and press E.")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
