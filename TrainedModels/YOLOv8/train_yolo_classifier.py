from pathlib import Path
from ultralytics import YOLO

# Este script debe ejecutarse desde la carpeta TrainedModels/YOLOv8/
BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = BASE_DIR / "data" / "images"  # contiene train/, val/, test/...
PROJECT_DIR = BASE_DIR / "runs"           # Ultralytics guardará aquí los resultados

def main():
    print("[INFO] Iniciando entrenamiento YOLOv8-cls para gestos de mano.")
    print(f"[INFO] Dataset de imágenes: {DATA_ROOT}")

    # Modelo base de clasificación
    model = YOLO("yolov8n-cls.pt")  # se descarga la primera vez

    model.train(
        data=str(DATA_ROOT),
        epochs=50,
        imgsz=224,
        project=str(PROJECT_DIR),
        name="hands_gestures",
        verbose=True,
    )

    print("[INFO] Entrenamiento terminado.")
    print(f"[INFO] Revisa los pesos en: {PROJECT_DIR / 'hands_gestures' / 'weights'}")


if __name__ == "__main__":
    main()
