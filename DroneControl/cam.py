import cv2
import numpy as np
import pickle
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================

MODEL_PATH = "./TrainedModels/MediaPipe/trained_model_mediapipe.pkl"
HAND_MODEL = "./TrainedModels/MediaPipe/hand_landmarker.task"

MAX_FRAMES = 48
NUM_LANDMARKS = 21
FEAT_DIM = NUM_LANDMARKS * 3

# ================= CARGAR MODELO =================

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
label_names = data["label_names"]

# ================= MEDIA PIPE =================

base_options = python.BaseOptions(model_asset_path=HAND_MODEL)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= HELPERS =================

def extract_features(X_seq):
    """
    X_seq: (48, 63)
    return: (252,)
    """
    return np.hstack([
        X_seq.mean(axis=0),
        X_seq.std(axis=0),
        X_seq.max(axis=0),
        X_seq.min(axis=0)
    ])

# ================= MAIN =================

sequence_buffer = deque(maxlen=MAX_FRAMES)
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)
        gesture = "..."

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            frame_feat = []

            for lm in hand:
                frame_feat.extend([lm.x, lm.y, lm.z])

            if len(frame_feat) == FEAT_DIM:
                sequence_buffer.append(frame_feat)

        if len(sequence_buffer) == MAX_FRAMES:
            X_seq = np.array(sequence_buffer, dtype=np.float32)
            X_feat = extract_features(X_seq).reshape(1, -1)
            X_feat = scaler.transform(X_feat)

            pred = model.predict(X_feat)[0]
            gesture = label_names[pred]

        # ================= UI =================
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
