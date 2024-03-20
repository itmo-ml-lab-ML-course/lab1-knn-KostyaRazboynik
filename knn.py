import math

import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNClassifier:
    def __init__(
            self,
            window_type: str = "mutable",
            window_size: float | int = 5,
            kernel: str = 'uniform',
            metric: str = 'manhattan',
            weight: list | None = None
    ):
        self._validate_args(kernel, metric, window_type)
        self.h: int | None = None
        self.n_neighbors: int | None = None
        if window_type == 'fixed':
            self.h = window_size
        else:
            self.n_neighbors = window_size
        self.kernel: str = kernel
        self.metric: str = metric
        self.class_count: int | None = None
        self.y_train = None
        self.X_train = None
        self.weights = weight

    def fit(self, X, y):
        if len(X) != len(y):
            raise RuntimeError('X and y with different lengths')
        self.X_train = X
        self.y_train = y
        self.class_count = len(set(y))
        self.weights = self.weights if self.weights is not None else np.ones(len(self.y_train))
        return self

    def predict(self, X):
        distances, indexes = self._count_distances_and_indexes(
            X=X,
            n_neighbors=self.n_neighbors + 1 if self.n_neighbors is not None else 5
        )

        classes, weights = self._count_classes_and_weights(indexes=indexes)
        prediction = []
        for obj in range(len(X)):
            dist, cls, w = distances[obj], classes[obj], weights[obj]
            scores = np.zeros(self.class_count).tolist()
            for i in range(len(dist)):
                radius = self.h if self.h is not None else dist[-1]
                scores[cls[i]] += self._count_kernel(dist[i] / radius) * w[i]
            prediction.append(scores.index(max(scores)))

        return prediction

    def _count_distances_and_indexes(self, X, n_neighbors):
        return (NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric.lower())
                .fit(self.X_train)
                .kneighbors(X, n_neighbors=n_neighbors))

    def _count_classes_and_weights(self, indexes):
        classes, weights = [], []
        for i in indexes:
            classes.append(self.y_train.iloc[i].to_list())
            weights.append([self.weights[j] for j in i])
        return classes, weights

    @staticmethod
    def _validate_args(kernel, metric, window_type):
        if kernel not in ['uniform', 'triangular', 'epanechnikov', 'gaussian']:
            raise RuntimeError('kernel must be uniform or triangular or epanechnikov or gaussian')
        if metric not in ['manhattan', 'euclidean', 'cosine']:
            raise RuntimeError('metric must be manhattan or euclidean or cosine')
        if window_type not in ['fixed', 'mutable']:
            raise RuntimeError('window_type must be fixed or mutable')

    def _count_kernel(self, x: float) -> float:
        match self.kernel:
            case 'uniform':
                return self._uniform_kernel(x)
            case 'epanechnikov':
                return self._epanechnikov_kernel(x)
            case 'triangular':
                return self._triangular_kernel(x)
            case 'gaussian':
                return self._gaussian_kernel(x)
            case _:
                return 0  # unreachable state because of _validate_args in __init__()

    @staticmethod
    def _uniform_kernel(x: float) -> float:
        return 0.5 if abs(x) <= 1 else 0

    @staticmethod
    def _epanechnikov_kernel(x: float) -> float:
        return 3 / 4 * (1 - int(x) ** 2)

    @staticmethod
    def _triangular_kernel(x: float) -> float:
        return 1 - abs(int(x)) if abs(x) <= 1 else 0

    @staticmethod
    def _gaussian_kernel(x: float) -> float:
        return 1 / math.sqrt(2 * math.pi) * math.exp((-x ** 2) / 2)
