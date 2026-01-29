import os
import numpy as np
import pickle
from scipy import interpolate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = SCRIPT_DIR

# ======================
# Data augmentation
# ======================
def augment_sequence(X_seq, n_augmentations=2):
    """Genera versiones aumentadas de una secuencia temporal"""
    augmented = []
    T, D = X_seq.shape
    
    for _ in range(n_augmentations):
        # Ruido gaussiano
        noise = np.random.normal(0, 0.02, X_seq.shape)
        augmented.append(X_seq + noise)
        
        # Time warping
        warp_factor = np.random.uniform(0.8, 1.2)
        new_length = max(2, int(T * warp_factor))
        new_indices = np.linspace(0, T-1, new_length)
        X_warped = np.zeros((new_length, D))
        for d in range(D):
            interp_func = interpolate.interp1d(np.arange(T), X_seq[:, d], kind='linear', fill_value='extrapolate')
            X_warped[:, d] = interp_func(new_indices)
        
        # Redimensionar a longitud original
        if new_length != T:
            final_indices = np.linspace(0, new_length-1, T)
            X_warped_resized = np.zeros((T, D))
            for d in range(D):
                interp_func = interpolate.interp1d(np.arange(new_length), X_warped[:, d], kind='linear', fill_value='extrapolate')
                X_warped_resized[:, d] = interp_func(final_indices)
            X_warped = X_warped_resized
        augmented.append(X_warped)
        
        # Escalado
        augmented.append(X_seq * np.random.uniform(0.9, 1.1))
        
        # Desplazamiento temporal
        augmented.append(np.roll(X_seq, np.random.randint(-2, 3), axis=0))
        
        # Rotación aleatoria 2D (XY)
        theta = np.random.uniform(-0.2, 0.2)  # +/- 0.2 rad
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X_rot = X_seq.copy()
        for i in range(0, D, 3):
            x, y = X_seq[:, i], X_seq[:, i+1]
            X_rot[:, i] = cos_t*x - sin_t*y
            X_rot[:, i+1] = sin_t*x + cos_t*y
        augmented.append(X_rot)
    
    return augmented

def augment_dataset(X, y, n_augmentations=2):
    """Aumenta todo el dataset"""
    X_aug_list = [X]
    y_aug_list = [y]
    
    for i in range(len(X)):
        augmented_seqs = augment_sequence(X[i], n_augmentations=n_augmentations)
        for aug_seq in augmented_seqs:
            X_aug_list.append(aug_seq[np.newaxis, :, :])
            y_aug_list.append(np.array([y[i]]))
    
    return np.vstack(X_aug_list), np.concatenate(y_aug_list)

# ======================
# Feature extraction
# ======================
def extract_angular_features(X):
    """Extrae características angulares para movimientos circulares"""
    angular_features_list = []
    
    for seq in X:
        T = seq.shape[0]
        n_landmarks = seq.shape[1] // 3
        
        x_center = seq[:, 0::3].mean(axis=1)
        y_center = seq[:, 1::3].mean(axis=1)
        
        angles = np.arctan2(np.diff(y_center), np.diff(x_center)) if T > 1 else np.array([0])
        angle_diffs = np.diff(angles) if len(angles) > 1 else np.array([0])
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        total_rotation = np.sum(angle_diffs)
        mean_velocity = np.mean(angle_diffs) if len(angle_diffs) > 0 else 0
        std_velocity = np.std(angle_diffs) if len(angle_diffs) > 0 else 0
        cumulative_angle = np.cumsum(angle_diffs)
        max_rotation = np.max(np.abs(cumulative_angle)) if len(cumulative_angle) > 0 else 0
        direction_consistency = np.sum(np.sign(angle_diffs) == np.sign(total_rotation)) / len(angle_diffs) if total_rotation != 0 else 0
        rotation_range = np.max(cumulative_angle) - np.min(cumulative_angle) if len(cumulative_angle) > 0 else 0
        direction_changes = np.sum(np.diff(np.sign(angle_diffs)) != 0) if len(angle_diffs) > 1 else 0
        
        angular_features_list.append([
            total_rotation,
            mean_velocity,
            std_velocity,
            max_rotation,
            direction_consistency,
            rotation_range,
            direction_changes
        ])
    
    return np.array(angular_features_list)

def extract_velocity_features(X):
    """Velocidad y aceleración"""
    velocity_features_list = []
    for seq in X:
        velocities = np.diff(seq, axis=0)
        if len(velocities) == 0:
            velocity_features_list.append([0,0,0,0])
            continue
        mean_velocity = np.mean(np.abs(velocities))
        max_velocity = np.max(np.abs(velocities))
        accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.zeros_like(velocities)
        mean_acceleration = np.mean(np.abs(accelerations))
        max_acceleration = np.max(np.abs(accelerations))
        velocity_features_list.append([mean_velocity, max_velocity, mean_acceleration, max_acceleration])
    return np.array(velocity_features_list)

def extract_distance_features(X):
    """Distancias relativas entre articulaciones"""
    distance_features_list = []
    for seq in X:
        D = seq.shape[1] // 3
        distances = []
        for i in range(D):
            for j in range(i+1, D):
                diff = seq[:, i*3:i*3+3] - seq[:, j*3:j*3+3]
                dist = np.linalg.norm(diff, axis=1)
                distances.append(dist.mean())
                distances.append(dist.std())
        distance_features_list.append(distances)
    return np.array(distance_features_list)

def extract_features(X):
    """Combina estadísticas, angulares, velocidad y distancias"""
    statistical_features = np.hstack([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1)
    ])
    angular_features = extract_angular_features(X)
    velocity_features = extract_velocity_features(X)
    distance_features = extract_distance_features(X)
    return np.hstack([statistical_features, angular_features, velocity_features, distance_features ])

# ======================
# Load dataset
# ======================
def load_split(name, apply_augmentation=False, n_aug=2):
    data = np.load(os.path.join(INPUT_DIR, name))
    X, y, labels = data["X"], data["y"], data["label_names"]
    
    if apply_augmentation:
        X, y = augment_dataset(X, y, n_augmentations=n_aug)
    
    return extract_features(X), y, labels

# ======================
# Entrenamiento y evaluación
# ======================
def main():
    # Cargar splits
    X_train, y_train, labels = load_split("mediapipe_train.npz", apply_augmentation=True, n_aug=3)
    X_val, y_val, _ = load_split("mediapipe_val.npz")
    X_test, y_test, _ = load_split("mediapipe_test.npz")

    # Escalado robusto
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Modelo RandomForest ajustado
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Evaluación
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy en validación: {val_acc:.3f}")
    print(f"Accuracy en test: {test_acc:.3f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, model.predict(X_test))
    out_cm = os.path.join(INPUT_DIR, "confusion_matrix_mediapipe.png")
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_cm, dpi=150)
    plt.close()

    # Guardar modelo
    model_path = os.path.join(INPUT_DIR, "trained_model_mediapipe.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "label_names": labels}, f)
    print(f"Matriz de confusión guardada en: {out_cm}")
    print(f"Modelo entrenado guardado en: {model_path}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
