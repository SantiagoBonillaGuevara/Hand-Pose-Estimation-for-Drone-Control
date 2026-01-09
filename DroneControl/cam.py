import cv2
import numpy as np
import pickle
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import interpolate

# ================= CONFIG =================

MODEL_PATH = "./TrainedModels/MediaPipe/trained_model_mediapipe.pkl"
HAND_MODEL = "./TrainedModels/MediaPipe/hand_landmarker.task"

MAX_FRAMES = 48
NUM_LANDMARKS = 21
FEAT_DIM = NUM_LANDMARKS * 3  # 63

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

# ================= FEATURE EXTRACTION =================

def extract_angular_features(X):
    angular_features_list = []

    for seq in X:
        T = seq.shape[0]
        n_landmarks = seq.shape[1] // 3

        x_center, y_center = [], []

        for t in range(T):
            x_coords = seq[t, 0::3]
            y_coords = seq[t, 1::3]
            x_center.append(np.mean(x_coords))
            y_center.append(np.mean(y_coords))

        x_center = np.array(x_center)
        y_center = np.array(y_center)

        angles = []
        for i in range(1, len(x_center)):
            dx = x_center[i] - x_center[i - 1]
            dy = y_center[i] - y_center[i - 1]
            angles.append(np.arctan2(dy, dx))

        if len(angles) < 2:
            angular_features_list.append([0, 0, 0, 0, 0, 0, 0])
            continue

        angles = np.array(angles)
        angle_diffs = np.diff(angles)
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))

        total_rotation = np.sum(angle_diffs)
        mean_angular_velocity = np.mean(angle_diffs)
        std_angular_velocity = np.std(angle_diffs)

        cumulative_angle = np.cumsum(angle_diffs)
        max_rotation = np.max(np.abs(cumulative_angle))

        direction_consistency = (
            np.sum(np.sign(angle_diffs) == np.sign(total_rotation)) / len(angle_diffs)
            if total_rotation != 0 else 0
        )

        rotation_range = np.max(cumulative_angle) - np.min(cumulative_angle)
        direction_changes = np.sum(np.diff(np.sign(angle_diffs)) != 0)

        angular_features_list.append([
            total_rotation,
            mean_angular_velocity,
            std_angular_velocity,
            max_rotation,
            direction_consistency,
            rotation_range,
            direction_changes
        ])

    return np.array(angular_features_list)


def extract_velocity_features(X):
    velocity_features_list = []

    for seq in X:
        velocities = np.diff(seq, axis=0)

        if len(velocities) == 0:
            velocity_features_list.append([0, 0, 0, 0])
            continue

        mean_velocity = np.mean(np.abs(velocities))
        max_velocity = np.max(np.abs(velocities))

        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            mean_acceleration = np.mean(np.abs(accelerations))
            max_acceleration = np.max(np.abs(accelerations))
        else:
            mean_acceleration = 0
            max_acceleration = 0

        velocity_features_list.append([
            mean_velocity,
            max_velocity,
            mean_acceleration,
            max_acceleration
        ])

    return np.array(velocity_features_list)


def extract_features(X):
    statistical_features = np.hstack([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1)
    ])

    angular_features = extract_angular_features(X)
    velocity_features = extract_velocity_features(X)

    return np.hstack([
        statistical_features,
        angular_features,
        velocity_features
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
            X_feat = extract_features(X_seq[np.newaxis, :, :])
            X_feat = scaler.transform(X_feat)

            probs = model.predict_proba(X_feat)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]

            gesture = label_names[best_idx] if confidence > 0.3 else "Unknown"
            
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