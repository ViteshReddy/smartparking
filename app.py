
from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime
import cv2
import threading

app = Flask(__name__)

OCCUPIED_THRESHOLD = 5.0

parking_areas = {
    "Area A": [
        {"id": 1, "occupied": False, "distance": None, "last_update": None},
        {"id": 2, "occupied": False, "distance": None, "last_update": None}
    ],
    "Area B": [
        {"id": 3, "occupied": False, "distance": None, "last_update": None},
        {"id": 4, "occupied": False, "distance": None, "last_update": None}
    ]
}

# Camera is opened only when the live preview is requested.
camera = None
camera_lock = threading.Lock()


def calculate_status():
    monitored = [
        slot
        for slots in parking_areas.values()
        for slot in slots
        if slot["id"] != 4
    ]

    occupied = sum(1 for slot in monitored if slot["occupied"])

    return {
        "total_slots": len(monitored),
        "occupied_slots": occupied,
        "available_slots": len(monitored) - occupied
    }


@app.route("/")
def dashboard():
    return render_template(
        "index.html",
        areas=parking_areas,
        **calculate_status()
    )


@app.route("/api/update", methods=["POST"])
def update_parking():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Expected JSON data"
        }), 400

    area = data.get("area")

    if area not in parking_areas:
        return jsonify({
            "success": False,
            "error": "Unknown parking area"
        }), 400

    readings = [data.get("slot1"), data.get("slot2")]
    updated_slots = []

    for slot, reading in zip(parking_areas[area], readings):
        try:
            distance = float(reading)
        except (TypeError, ValueError):
            continue

        # -1/no echo and invalid values must not overwrite valid readings.
        if not 0 < distance < 400:
            continue

        slot["distance"] = round(distance, 2)
        slot["occupied"] = distance <= OCCUPIED_THRESHOLD
        slot["last_update"] = datetime.now().isoformat(
            timespec="seconds"
        )
        updated_slots.append(slot["id"])

    return jsonify({
        "success": True,
        "area": area,
        "updated_slots": updated_slots,
        "slots": parking_areas[area],
        "threshold_cm": OCCUPIED_THRESHOLD,
        **calculate_status()
    })


@app.route("/api/status", methods=["GET"])
def parking_status():
    return jsonify({
        "areas": parking_areas,
        "threshold_cm": OCCUPIED_THRESHOLD,
        **calculate_status()
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "server": "ParkSync",
        "status": "running",
        "threshold_cm": OCCUPIED_THRESHOLD
    })


# ---------------- LIVE CAMERA ----------------

def get_camera():
    global camera

    with camera_lock:
        if camera is None or not camera.isOpened():
            if camera is not None:
                camera.release()

            camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if not camera.isOpened():
                camera.release()
                camera = None
                return None

        return camera


def camera_frames():
    global camera

    try:
        while True:
            with camera_lock:
                active_camera = get_camera()

                if active_camera is None:
                    return

                success, frame = active_camera.read()

                if not success:
                    return

                frame = cv2.resize(frame, (640, 360))

                encoded, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )

            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        # The feed ends when the browser closes the preview.
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None


@app.route("/camera_feed")
def camera_feed():
    return Response(
        camera_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )
