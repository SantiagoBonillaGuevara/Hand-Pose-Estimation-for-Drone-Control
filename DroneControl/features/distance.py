import numpy as np

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