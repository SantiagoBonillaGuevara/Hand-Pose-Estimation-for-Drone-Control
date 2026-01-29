import numpy as np


def extract_angular_features(X):
    features = []

    for seq in X:
        x_center = np.mean(seq[:, 0::3], axis=1)
        y_center = np.mean(seq[:, 1::3], axis=1)

        dx = np.diff(x_center)
        dy = np.diff(y_center)
        angles = np.arctan2(dy, dx)

        if len(angles) < 2:
            features.append([0] * 7)
            continue

        angle_diffs = np.diff(angles)
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))

        total_rotation = np.sum(angle_diffs)
        cumulative = np.cumsum(angle_diffs)

        features.append([
            total_rotation,
            np.mean(angle_diffs),
            np.std(angle_diffs),
            np.max(np.abs(cumulative)),
            np.mean(np.sign(angle_diffs) == np.sign(total_rotation))
            if total_rotation != 0 else 0,
            np.max(cumulative) - np.min(cumulative),
            np.sum(np.diff(np.sign(angle_diffs)) != 0)
        ])

    return np.array(features)
