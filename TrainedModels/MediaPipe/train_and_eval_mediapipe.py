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
            interp_func = interpolate.interp1d(np.arange(T), X_seq[:, d], 
                                               kind='linear', fill_value='extrapolate')
            X_warped[:, d] = interp_func(new_indices)
        
        # Redimensionar a longitud original
        if new_length != T:
            final_indices = np.linspace(0, new_length-1, T)
            X_warped_resized = np.zeros((T, D))
            for d in range(D):
                interp_func = interpolate.interp1d(np.arange(new_length), X_warped[:, d],
                                                   kind='linear', fill_value='extrapolate')
                X_warped_resized[:, d] = interp_func(final_indices)
            X_warped = X_warped_resized
        
        augmented.append(X_warped)
        
        # Escalado
        augmented.append(X_seq * np.random.uniform(0.9, 1.1))
        
        # Desplazamiento temporal
        augmented.append(np.roll(X_seq, np.random.randint(-2, 3), axis=0))
    
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

def extract_angular_features(X):
    """
    Extrae características angulares para detectar dirección de movimiento circular.
    CRÍTICO para distinguir clockwise vs counterclockwise.
    """
    angular_features_list = []
    
    for seq in X:
        # seq.shape = (timesteps, features)
        # Calculamos el centro de masa de la mano en cada frame
        T = seq.shape[0]
        
        # Extraer coordenadas x, y del centro de la mano
        # Asumimos que las features están organizadas como grupos de coordenadas (x, y, z)
        # Tomamos cada tercera coordenada para x e y
        n_landmarks = seq.shape[1] // 3
        
        # Calcular centro de masa en cada timestep
        x_center = []
        y_center = []
        
        for t in range(T):
            x_coords = seq[t, 0::3]  # Cada tercera coordenada empezando en 0 (x)
            y_coords = seq[t, 1::3]  # Cada tercera coordenada empezando en 1 (y)
            x_center.append(np.mean(x_coords))
            y_center.append(np.mean(y_coords))
        
        x_center = np.array(x_center)
        y_center = np.array(y_center)
        
        # Calcular ángulos entre frames consecutivos
        angles = []
        for i in range(1, len(x_center)):
            dx = x_center[i] - x_center[i-1]
            dy = y_center[i] - y_center[i-1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        if len(angles) < 2:
            # Si hay muy pocos frames, usar valores por defecto
            angular_features_list.append([0, 0, 0, 0, 0, 0, 0])
            continue
        
        angles = np.array(angles)
        
        # Calcular cambios angulares
        angle_diffs = np.diff(angles)
        # Normalizar ángulos a [-pi, pi]
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        # FEATURE 1: Rotación total acumulada (+ = CCW, - = CW)
        total_rotation = np.sum(angle_diffs)
        
        # FEATURE 2: Velocidad angular promedio
        mean_angular_velocity = np.mean(angle_diffs) if len(angle_diffs) > 0 else 0
        
        # FEATURE 3: Desviación estándar de velocidad angular
        std_angular_velocity = np.std(angle_diffs) if len(angle_diffs) > 0 else 0
        
        # FEATURE 4: Rotación acumulada máxima
        cumulative_angle = np.cumsum(angle_diffs)
        max_rotation = np.max(np.abs(cumulative_angle)) if len(cumulative_angle) > 0 else 0
        
        # FEATURE 5: Consistencia de dirección (qué tan consistente es el sentido de giro)
        if len(angle_diffs) > 0 and total_rotation != 0:
            direction_consistency = np.sum(np.sign(angle_diffs) == np.sign(total_rotation)) / len(angle_diffs)
        else:
            direction_consistency = 0
        
        # FEATURE 6: Rango de rotación (diferencia entre max y min)
        rotation_range = np.max(cumulative_angle) - np.min(cumulative_angle) if len(cumulative_angle) > 0 else 0
        
        # FEATURE 7: Número de cambios de dirección (cuántas veces cambia el signo)
        direction_changes = np.sum(np.diff(np.sign(angle_diffs)) != 0) if len(angle_diffs) > 1 else 0
        
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
    """
    Extrae características de velocidad y aceleración.
    """
    velocity_features_list = []
    
    for seq in X:
        # Calcular velocidades (diferencias entre frames)
        velocities = np.diff(seq, axis=0)
        
        if len(velocities) == 0:
            velocity_features_list.append([0, 0, 0, 0])
            continue
        
        # Velocidad promedio
        mean_velocity = np.mean(np.abs(velocities))
        
        # Velocidad máxima
        max_velocity = np.max(np.abs(velocities))
        
        # Aceleración (cambio de velocidad)
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
    """
    Extrae features combinadas: estadísticas + angulares + velocidad
    """
    # Features estadísticas originales
    statistical_features = np.hstack([
        X.mean(axis=1), 
        X.std(axis=1), 
        X.max(axis=1), 
        X.min(axis=1)
    ])
    
    # Features angulares (NUEVAS - MÁS IMPORTANTES)
    angular_features = extract_angular_features(X)
    
    # Features de velocidad (NUEVAS)
    velocity_features = extract_velocity_features(X)
    
    # Combinar todas las features
    return np.hstack([statistical_features, angular_features, velocity_features])

def load_split(name, apply_augmentation=False, n_aug=2):
    """Carga un split del dataset"""
    data = np.load(os.path.join(INPUT_DIR, name))
    X, y, labels = data["X"], data["y"], data["label_names"]
    
    if apply_augmentation:
        X, y = augment_dataset(X, y, n_augmentations=n_aug)
    
    return extract_features(X), y, labels

def main():
    X_train, y_train, labels = load_split(
        "mediapipe_train.npz",
        apply_augmentation=True,
        n_aug=3
    )

    X_val, y_val, _ = load_split("mediapipe_val.npz")
    X_test, y_test, _ = load_split("mediapipe_test.npz")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Accuracy en validación: {val_acc:.3f}")
    print(f"Accuracy en test: {test_acc:.3f}")

    cm = confusion_matrix(y_test, model.predict(X_test))
    out_cm = os.path.join(INPUT_DIR, "confusion_matrix_mediapipe.png")

    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_cm, dpi=150)
    plt.close()

    model_path = os.path.join(INPUT_DIR, "trained_model_mediapipe.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": model, "scaler": scaler, "label_names": labels},
            f
        )

    print(f"Matriz de confusión guardada en: {out_cm}")
    print(f"Modelo entrenado guardado en: {model_path}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()