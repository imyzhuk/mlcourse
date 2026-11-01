import numpy as np
import pandas as pd

def calc_minkowski_distance(mtx, p):
    return np.pow(np.sum(np.pow(mtx, p), axis=1), 1/p)

class MyKNeighborsClassifier:
    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="brute",
        metric="minkowski",
        p=2
    ):
        if weights not in ["uniform", "distance"]:
            raise ValueError("'weights' must be uniform or distance!")

        if metric != "minkowski":
            raise ValueError("'metric' must be only 'minkowski'!")
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.src_X = None
        self.src_y = None

    def fit(self, X, y):
        self.src_X = X.copy()
        self.src_y = y.copy()
        return self

    def predict(self, X):
        if self.weights == "uniform":
            return self._vote_on_quantity(X)

        return self._vote_on_distance(X)

    def _vote_on_quantity(self, X):
        src_X, src_y, p, n_neighbors = self.src_X, self.src_y, self.p, self.n_neighbors

        predictions = []

        for _, curr_sample in X.iterrows():
            distances_mtx = np.abs(src_X - curr_sample)
            
            distances = calc_minkowski_distance(distances_mtx, p)
            nearest_sample_indexes = np.argsort(distances)[:n_neighbors]
            nearest_values = src_y.iloc[nearest_sample_indexes]
            
            values, counts = np.unique(nearest_values, return_counts=True)
            if len(counts[counts == counts.max()]):
                values[np.argmax(counts)]
            
            max_nearest_classes = values[counts == counts.max()]
            
            nearest_distances = distances.iloc[nearest_sample_indexes]
            nearest_distances.index = nearest_values
            nearest_distances = nearest_distances[nearest_distances.index.isin(max_nearest_classes)]

            predictions.append(nearest_distances.groupby(level=0).mean().idxmin())

        return predictions

    def _vote_on_distance(self, X):
        src_X, src_y, p, n_neighbors = self.src_X, self.src_y, self.p, self.n_neighbors

        predictions = []

        for _, curr_sample in X.iterrows():
            distances_mtx = np.abs(src_X - curr_sample)

            distances = calc_minkowski_distance(distances_mtx, p)
            nearest_sample_indexes = np.argsort(distances)[:n_neighbors]

            nearest_values = src_y.iloc[nearest_sample_indexes]
            nearest_distances = distances.iloc[nearest_sample_indexes]
            nearest_distances.index = nearest_values
            
            predictions.append(nearest_distances.groupby(level=0).mean().idxmin())

        return predictions