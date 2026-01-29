import numpy as np


def extract_velocity_features(X):
    features = []

    for seq in X:
        velocities = np.diff(seq, axis=0)

        if len(velocities) == 0:
            features.append([0, 0, 0, 0])
            continue

        mean_v = np.mean(np.abs(velocities))
        max_v = np.max(np.abs(velocities))

        acc = np.diff(velocities, axis=0) if len(velocities) > 1 else []
        mean_a = np.mean(np.abs(acc)) if len(acc) else 0
        max_a = np.max(np.abs(acc)) if len(acc) else 0

        features.append([mean_v, max_v, mean_a, max_a])

    return np.array(features)
