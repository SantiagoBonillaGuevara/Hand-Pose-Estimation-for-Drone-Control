import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==================== RUTAS BASADAS EN EL SCRIPT ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR   # los NPZ están en el mismo directorio

def load_split(name):
    path = os.path.join(INPUT_DIR, name)
    data = np.load(path)
    X = data["X"]
    y = data["y"]

    # Si el archivo tiene label_names, los usamos; si no, los inferimos
    if "label_names" in data:
        labels = data["label_names"]
    else:
        unique = np.unique(y)
        # Los convertimos a strings por si luego quieres imprimirlos bonitos
        labels = np.array([str(l) for l in unique])

    # X puede venir como (N, T, D) o como (N, D)
    if X.ndim == 3:
        # Reducir secuencia: promedio sobre el tiempo
        X_flat = X.mean(axis=1)   # (N, D)
    else:
        X_flat = X  # ya está plano

    return X_flat, y, labels

def main():
    print("Cargando datos desde:", INPUT_DIR)

    # === Train, Val y test (requeridos) ===
    X_train, y_train, label_names = load_split("openpose_train.npz")
    X_val,   y_val,   _           = load_split("openpose_val.npz")
    X_test,  y_test,  _           = load_split("openpose_test.npz")

    # Un modelo sencillo, igual estilo que MediaPipe
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=200,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Validación
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy en validación: {val_acc:.3f}")

    # Test final
    y_eval_pred = clf.predict(X_test)
    eval_acc = accuracy_score(y_test, y_eval_pred)
    print(f"Accuracy en test: {eval_acc:.3f}")

    cm = confusion_matrix(y_test, y_eval_pred)
    cm_filename = "confusion_matrix_openpose_test.png"
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    # Guardar matriz en el MISMO directorio del script
    out_path = os.path.join(INPUT_DIR, cm_filename)
    plt.savefig(out_path)
    print(f"Matriz de confusión guardada en: {out_path}")

if __name__ == "__main__":
    main()
