import numpy as np
from .angular import extract_angular_features
from .velocity import extract_velocity_features
from .distance import extract_distance_features


def extract_features(X):
    stats = np.hstack([
        X.mean(axis=1),
        X.std(axis=1),
        X.max(axis=1),
        X.min(axis=1)
    ])

    angular = extract_angular_features(X)
    velocity = extract_velocity_features(X)
    distance = extract_distance_features(X)

    return np.hstack([stats, angular, velocity, distance])
