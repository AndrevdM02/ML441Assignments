import numpy as np # type: ignore
from scipy.spatial import distance # type: ignore
from tqdm import tqdm # type: ignore
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import mode # type: ignore

class KNN:
    def __init__(self, n_neighbors=5, p=2, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.classes_ = None
        self.effective_metric_ = None
        self.n_samples_fit_ = None
        self.n_features_in_ = None
        self.X_ = None
        self.y_ = None
        self.neighbors_indices = None
        self.neighbors_distances = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.p == 2:
            self.effective_metric_ = "euclidean"
        elif self.p == 1:
            self.effective_metric_ = "manhattan"
        else:
            self.effective_metric_ = "minkowski"
        self.n_samples_fit_, self.n_features_in_ = X.shape
        self.X_ = X
        self.y_ = y

        return self

    def _compute_distance(self, X_row, X_db_row):
        distance_ = 0
        pw = self.p
        for i in range(self.n_features_in_):
            if not (np.isnan(X_row[i]) or np.isnan(X_db_row[i])):
                distance_ += np.abs(X_row[i] - X_db_row[i]) ** pw
        distance_ = distance_ ** (1/pw)
        return distance_

    def compute_row_distances(self, X_row, X_db):
        x_dist = []
        for X_db_row in X_db:
            distance_ = self._compute_distance(X_row, X_db_row)
            x_dist.append(distance_)
        return np.asarray(x_dist)

    def _compute_distances(self, X):
        dist = np.zeros((X.shape[0], len(self.X_)))

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.compute_row_distances, X[x], self.X_) for x in range(X.shape[0])]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Distances"):
                x_index = futures.index(future)
                dist[x_index] = future.result()

        return dist
    
    def kneighbors(self, X=None):
        if X is None:
            X = self.X_

        dist = self._compute_distances(X)
        neighbors_indices = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        neighbors_distances = np.sort(dist, axis=1)[:, :self.n_neighbors]
        
        return neighbors_indices, neighbors_distances

    def predict(self, X):
        dist = self._compute_distances(X)
        self.neighbors_indices = np.argsort(dist, axis=1)[:, :self.n_neighbors]
        self.neighbors_distances = np.sort(dist, axis=1)[:, :self.n_neighbors]

        # Initialize the list to hold the predicted class labels
        y_pred = []

        # For each test sample, find the nearest neighbors
        for i in range(dist.shape[0]):
            # Get the indices of the k-nearest neighbors
            nearest_indices = np.argsort(dist[i])[:self.n_neighbors]
            nearest_distances = dist[i][nearest_indices]
            nearest_labels = self.y_[nearest_indices]
            
            if self.weights == "uniform":
                # Uniform weights: each neighbor has the same weight
                weights = np.ones(self.n_neighbors)
            elif self.weights == "distance":
                # Distance-based weights: closer neighbors have more weight
                weights = 1 / (nearest_distances + 1e-5)  # Adding a small constant to avoid division by zero
            else:
                raise ValueError("Invalid weight type. Use 'uniform' or 'distance'.")

            # Compute weighted vote
            weighted_votes = {}
            for label, weight in zip(nearest_labels, weights):
                label = label.item() if isinstance(label, np.ndarray) else label
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight

            # Determine the most frequent (or highest weighted) class label
            most_common_label = max(weighted_votes, key=weighted_votes.get)
            y_pred.append(most_common_label)

        return np.array(y_pred)