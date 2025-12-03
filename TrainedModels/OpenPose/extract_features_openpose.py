import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==================== PATHS BASADOS EN EL SCRIPT ====================

# Directorio donde está este script: .../TrainedModels/OpenPose
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Raíz del repo: subimos dos niveles desde OpenPose -> TrainedModels -> raíz
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Carpeta CollectedData en la raíz del repo
DATA_DIR = os.path.join(PROJECT_ROOT, "CollectedData")

# CSV del dataset (mismo que MediaPipe)
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")
VIDEOS_BASE_DIR = DATA_DIR  # aquí están ThumbsUp/, Point/, etc.

# Carpeta de salida = mismo directorio donde está el script
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SCRIPT_DIR:", SCRIPT_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("CSV_PATH:", CSV_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)

# ==================== CONFIG OPENPOSE ====================

# Ruta a OpenPose Python API
# Ajusta esto según donde tengas tu build/python/
OPENPOSE_PATH = "/usr/local/python"
sys.path.append(OPENPOSE_PATH)

from openpose import pyopenpose as op

FRAME_STRIDE = 1

params = dict()
params["model_folder"] = os.path.join(PROJECT_ROOT, "models")  # AJUSTA si tus modelos están en otro lado
params["hand"] = True
params["hand_detector"] = 2      # usa detector interno
params["number_people_max"] = 1  # solo una persona

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def extract_hand_keypoints_from_video(video_path):
    """
    Extrae 21 keypoints de la mano derecha por frame y devuelve un vector (63,) promedio temporal.
    """
    cap = cv2.VideoCapture(video_path)
    frames_features = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Mano derecha = index 1
        if datum.handKeypoints[1].size == 0:
            frame_idx += 1
            continue

        hand_kp = datum.handKeypoints[1][0]  # shape: (21,3)
        frames_features.append(hand_kp.reshape(-1))  # (63,)

        frame_idx += 1

    cap.release()

    if len(frames_features) == 0:
        # fallback si no detecta mano
        return np.zeros(63, dtype=np.float32)

    frames_features = np.array(frames_features)  # (T, 63)
    return frames_features.mean(axis=0)          # (63,)


def main():
    df = pd.read_csv(CSV_PATH)

    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando videos con OpenPose"):
        rel_path = row["path"]          # p.ej. "ThumbsUp/thumbsUp_frontAngle.mp4"
        video_path = os.path.join(VIDEOS_BASE_DIR, rel_path)

        features = extract_hand_keypoints_from_video(video_path)

        split = row["split"]
        pose_id = row["pose_id"]

        if split == "train":
            X_train.append(features)
            y_train.append(pose_id)
        elif split == "val":
            X_val.append(features)
            y_val.append(pose_id)
        elif split == "test":
            X_test.append(features)
            y_test.append(pose_id)

    # Convertir a arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_val   = np.array(X_val,   dtype=np.float32)
    y_val   = np.array(y_val,   dtype=np.int64)
    X_test  = np.array(X_test,  dtype=np.float32)
    y_test  = np.array(y_test,  dtype=np.int64)

    # Guardar 3 splits en el MISMO directorio
    np.savez(os.path.join(OUTPUT_DIR, "openpose_train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(OUTPUT_DIR, "openpose_val.npz"),   X=X_val,   y=y_val)
    np.savez(os.path.join(OUTPUT_DIR, "openpose_test.npz"),  X=X_test,  y=y_test)

    print("\n✔ Listo:")
    print("  openpose_train.npz")
    print("  openpose_val.npz")
    print("  openpose_test.npz")
    print("Guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
