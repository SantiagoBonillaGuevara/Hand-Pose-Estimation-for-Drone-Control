import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==================== PATHS BASADOS EN EL SCRIPT ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "CollectedData")

# ==================== CONFIG ====================

CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
VIDEOS_BASE_DIR = DATA_DIR

MAX_FRAMES = 48
NUM_LANDMARKS = 21
FEAT_DIM = NUM_LANDMARKS * 3
FRAME_STRIDE = 1

OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SCRIPT_DIR:", SCRIPT_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("CSV_PATH:", CSV_PATH)

# ==================== HELPERS ====================

def sample_or_pad(sequence, max_frames, feat_dim):
    seq = np.array(sequence, dtype=np.float32)
    num_frames = seq.shape[0]

    if num_frames == 0:
        return np.zeros((max_frames, feat_dim), dtype=np.float32)

    if num_frames >= max_frames:
        indices = np.linspace(0, num_frames - 1, max_frames).astype(int)
        return seq[indices]
    else:
        pad_len = max_frames - num_frames
        pad = np.zeros((pad_len, feat_dim), dtype=np.float32)
        return np.vstack([seq, pad])


def extract_sequence_from_video(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir el video: {video_path}")
        return np.zeros((MAX_FRAMES, FEAT_DIM), dtype=np.float32)

    frame_idx = 0
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        # Convertir frame a formato MediaPipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detectar landmarks
        detection_result = landmarker.detect(mp_image)

        # Verificar si hay manos detectadas
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            features = []
            for landmark in hand_landmarks:
                features.extend([landmark.x, landmark.y, landmark.z])
            if len(features) == FEAT_DIM:
                sequence.append(features)

        frame_idx += 1

    cap.release()
    return sample_or_pad(sequence, MAX_FRAMES, FEAT_DIM)

# ==================== MAIN ====================

def main():
    df = pd.read_csv(CSV_PATH)

    unique_poses = sorted(df["pose"].unique())
    pose_to_id = {pose: i for i, pose in enumerate(unique_poses)}

    print("Gestos encontrados y sus IDs:")
    for pose, idx in pose_to_id.items():
        print(f"  {idx}: {pose}")

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Configurar HandLandmarker con Tasks API
    base_options = python.BaseOptions(
        model_asset_path='hand_landmarker.task'  # Descarga este modelo
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for row in df.itertuples():
            rel_path = row.path
            pose = row.pose
            split = row.split

            label_id = pose_to_id[pose]
            video_path = os.path.join(VIDEOS_BASE_DIR, rel_path)

            print(f"Procesando: {video_path}  | pose={pose}  | split={split}")

            seq = extract_sequence_from_video(video_path, landmarker)

            if split == "train":
                X_train.append(seq)
                y_train.append(label_id)
            elif split == "val":
                X_val.append(seq)
                y_val.append(label_id)
            elif split == "test":
                X_test.append(seq)
                y_test.append(label_id)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    label_names = np.array(unique_poses)

    print("Shapes finales:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:",   X_val.shape,   "y_val:",   y_val.shape)
    print("  X_test:",  X_test.shape,  "y_test:",  y_test.shape)

    np.savez(os.path.join(OUTPUT_DIR, "mediapipe_train.npz"),
             X=X_train, y=y_train, label_names=label_names)

    np.savez(os.path.join(OUTPUT_DIR, "mediapipe_val.npz"),
             X=X_val, y=y_val, label_names=label_names)

    np.savez(os.path.join(OUTPUT_DIR, "mediapipe_test.npz"),
             X=X_test, y=y_test, label_names=label_names)

    print(f"Archivos guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()