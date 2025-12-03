import os
from pathlib import Path
import glob

import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# ===================== CONFIGURACIÓN =====================

BASE_DIR = Path(__file__).resolve().parent

# Dataset a evaluar: puedes cambiar "val" por "test" si lo usas
SPLIT_TO_EVAL = "val"

IMAGES_SPLIT_DIR = BASE_DIR / "data" / "images" / SPLIT_TO_EVAL

# Ruta del modelo entrenado (ajústala según lo que te genere Ultralytics)
RUNS_DIR = BASE_DIR / "runs" / "hands_gestures"
WEIGHTS_PATH = RUNS_DIR / "weights" / "best.pt"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_image_paths_and_labels(root_dir: Path):
    """
    Estructura esperada:
      root_dir/
        ├─ CircleClockwise/*.jpg
        ├─ CircleCounterclockwise/*.jpg
        └─ ...

    Devuelve:
      paths: lista con rutas a imágenes
      labels: lista de índices de clase
      class_names: lista de nombres de clase (ordenados alfabéticamente)
    """
    class_names = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    image_paths = []
    labels = []

    for cls_name in class_names:
        cls_dir = root_dir / cls_name
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in cls_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(class_to_idx[cls_name])

    return image_paths, np.array(labels), class_names


def main():
    print(f"[INFO] Cargando modelo desde: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    print(f"[INFO] Cargando imágenes de: {IMAGES_SPLIT_DIR}")
    image_paths, y_true, class_names = load_image_paths_and_labels(IMAGES_SPLIT_DIR)

    print(f"[INFO] Nº de imágenes en {SPLIT_TO_EVAL}: {len(image_paths)}")
    print(f"[INFO] Clases: {class_names}")

    y_pred = []

    for img_path in image_paths:
        results = model(img_path)
        r = results[0]

        # Para clasificación, r.probs.top1 es el índice de clase predicha
        pred_idx = int(r.probs.top1)
        y_pred.append(pred_idx)

    y_pred = np.array(y_pred)

    # ================= Matriz de confusión =================
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)

    ax.set_title(f"Confusion Matrix - YOLOv8 ({SPLIT_TO_EVAL})")
    plt.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix_yolov8.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    print(f"[INFO] Matriz de confusión guardada en: {cm_path}")

    # ================= Reporte de clasificación =================
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report - YOLOv8\n")
        f.write(f"Split evaluado: {SPLIT_TO_EVAL}\n\n")
        f.write(report)

    print("[INFO] Reporte de clasificación:")
    print(report)
    print(f"[INFO] Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
