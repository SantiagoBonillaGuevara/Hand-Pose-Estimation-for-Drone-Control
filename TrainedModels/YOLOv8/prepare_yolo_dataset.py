import os
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ===================== CONFIGURACIÓN DE RUTAS =====================

# Este archivo está en TrainedModels/YOLOv8/
BASE_DIR = Path(__file__).resolve().parent

# Raíz del proyecto: subes dos niveles (YOLOv8 -> TrainedModels -> raíz)
PROJECT_ROOT = BASE_DIR.parent.parent

# Carpeta con videos y dataset.csv
COLLECTED_DIR = PROJECT_ROOT / "CollectedData"
CSV_PATH = COLLECTED_DIR / "dataset.csv"

# Salida de imágenes para clasificación
IMAGES_ROOT = BASE_DIR / "data" / "images"

# Modelo de detección de manos (ajusta esta ruta a tu modelo real)
# Puedes empezar con un modelo general de YOLOv8 y luego mejorarlo.
YOLO_MODEL_PATH = "yolov8n.pt"  # <-- CÁMBIALO si tienes un modelo de manos

FRAME_STRIDE = 3  # procesar 1 de cada 3 frames para no explotar el dataset(inicialmente 3)


def main():
    print(f"[INFO] Leyendo CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print(f"[INFO] Cargando modelo YOLO: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    for i, row in df.iterrows():
        rel_path = row["path"]           # p.ej. 'ThumbsUp/thumbsUp_frontAngle.mp4'
        pose_name = str(row["pose"])     # p.ej. 'ThumbsUp'
        split = str(row["split"])        # 'train' / 'val' / 'test'
        condition = str(row["condition"])
        index = str(row["index"])

        video_path = COLLECTED_DIR / rel_path
        if not video_path.exists():
            print(f"[WARN] Video no encontrado: {video_path}")
            continue

        out_dir = IMAGES_ROOT / split / pose_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Procesando video {video_path} -> {out_dir}")

        cap = cv2.VideoCapture(str(video_path))
        frame_id = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % FRAME_STRIDE != 0:
                continue

            # Inferencia YOLO
            results = model(frame)
            if len(results) == 0:
                continue

            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Tomar la caja más grande
            boxes = r.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_idx = areas.argmax()
            x1, y1, x2, y2 = boxes[max_idx].astype(int)

            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            filename = f"{pose_name}_idx{index}_{condition}_f{frame_id:04d}.jpg"
            out_path = out_dir / filename

            cv2.imwrite(str(out_path), crop)
            saved_count += 1

        cap.release()
        print(f"[INFO] Terminó {video_path}, guardados {saved_count} crops.")

    print("[INFO] Dataset de imágenes listo en:", IMAGES_ROOT)


if __name__ == "__main__":
    main()
