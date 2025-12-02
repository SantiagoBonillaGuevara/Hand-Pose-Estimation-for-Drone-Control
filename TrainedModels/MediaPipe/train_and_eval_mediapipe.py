import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ==================== RUTAS BASADAS EN EL SCRIPT ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR   # los NPZ están en el mismo directorio

def load_split(name):
    data = np.load(os.path.join(INPUT_DIR, name))
    X = data["X"]      # shape: (N, T, D)
    y = data["y"]
    labels = data["label_names"]
    # Reducir secuencia: promedio sobre el tiempo
    X_flat = X.mean(axis=1)   # (N, D)
    return X_flat, y, labels

def main():
    print("Cargando datos desde:", INPUT_DIR)

    X_train, y_train, label_names = load_split("mediapipe_train.npz")
    X_val,   y_val,   _          = load_split("mediapipe_val.npz")
    X_test,  y_test,  _          = load_split("mediapipe_test.npz")

    # Un modelo sencillo
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
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy en test: {test_acc:.3f}")

    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    # Guardar matriz en el MISMO directorio del script
    out_path = os.path.join(INPUT_DIR, "confusion_matrix_mediapipe.png")
    plt.savefig(out_path)
    print(f"Matriz de confusión guardada en: {out_path}")

if __name__ == "__main__":
    main()
