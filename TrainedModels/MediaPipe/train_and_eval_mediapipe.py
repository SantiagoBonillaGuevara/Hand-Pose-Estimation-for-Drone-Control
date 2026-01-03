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

def extract_features(X):
    """Extrae features estadísticas: mean, std, max, min"""
    return np.hstack([X.mean(axis=1), X.std(axis=1), X.max(axis=1), X.min(axis=1)])

def load_split(name, apply_augmentation=False, n_aug=2):
    """Carga un split del dataset"""
    data = np.load(os.path.join(INPUT_DIR, name))
    X, y, labels = data["X"], data["y"], data["label_names"]
    
    if apply_augmentation:
        X, y = augment_dataset(X, y, n_augmentations=n_aug)
    
    return extract_features(X), y, labels

def main():
    print("Cargando datos desde:", INPUT_DIR)
    
    # Cargar datos
    X_train, y_train, label_names = load_split("mediapipe_train.npz", apply_augmentation=True, n_aug=2)
    X_val, y_val, _ = load_split("mediapipe_val.npz")
    X_test, y_test, _ = load_split("mediapipe_test.npz")
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar Random Forest
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        max_features='log2',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluar
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    print(f"Accuracy en validación: {val_acc:.3f}")
    print(f"Accuracy en test: {test_acc:.3f}")
    
    # Generar matriz de confusión
    y_test_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=45, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.3f}')
    plt.tight_layout()
    
    out_path = os.path.join(INPUT_DIR, "confusion_matrix_mediapipe.png")
    plt.savefig(out_path, dpi=150)
    print(f"Matriz de confusión guardada en: {out_path}")
    
    # Guardar modelo entrenado
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_names': label_names
    }
    model_path = os.path.join(INPUT_DIR, "trained_model_mediapipe.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Modelo entrenado guardado en: {model_path}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()