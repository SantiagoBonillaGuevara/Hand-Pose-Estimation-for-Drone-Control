import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ================= CONFIG =================

MODEL_PATH = "./TrainedModels/YOLOv8/runs/hands_gestures/weights/best.pt

CONF_THRESHOLD = 0.3
IMG_SIZE = 224

# ================= LOAD MODEL =================

model = YOLO(str(MODEL_PATH))
label_names = model.names

# ================= MAIN =================

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    gesture = "Unknown"
    confidence = 0.0

    if results and results[0].probs is not None:
        probs = results[0].probs
        idx = int(probs.top1)
        confidence = float(probs.top1conf)
        if confidence > CONF_THRESHOLD:
            gesture = label_names[idx]

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()