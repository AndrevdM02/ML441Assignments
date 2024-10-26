import numpy as np
from joblib import Parallel, delayed

class LogisticRegressionEnsemble:
    def __init__(self, num_models = 5, learning_rate = 0.1, iterations = 1000, patience = 5, bias = 1, bag_size=0.8, n_jobs=-1, num_features=0.8, random_state=None, polynomial_degree=1, num_nonlinear_features = 1):
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.bias = bias
        self.random_state = random_state
        self.iterations = iterations
        self.patience = patience
        self.bag_size = bag_size
        self.num_features = num_features
        self.polynomial_degree = polynomial_degree
        self.models = []
        self.n_jobs = n_jobs
        self.validation_error = []
        self.num_nonlinear_features = num_nonlinear_features
        self.feature_nonlinear_index = []
    
    def sigmoid(self, z):
        return np.clip(1 / (1 + np.exp(-np.clip(z, -709, 709))), 1e-12, 1 - 1e-12)
    
    def init_weights(self, num_features, iter):
        if self.random_state is not None:
            np.random.seed(self.random_state + iter)
        bounds = 1 / np.sqrt(num_features)
        return np.random.uniform(-bounds, bounds, (num_features))
    
    def generate_polynomial_terms(self, n_features):
        if n_features < 1:
            return []
        
        feature_indices = range(n_features)
        terms = []
        
        for i in feature_indices:
            terms.append([i])
            
        # Add higher degree terms
        for degree in range(2, self.polynomial_degree + 1):
            for i in feature_indices:
                terms.append([i] * degree)
                
            # Interaction terms
            if degree > 1:
                for i in feature_indices:
                    for j in feature_indices[i+1:]:  # Only consider unique combinations
                        term = [i] * (degree - 1) + [j]
                        terms.append(term)
                        if degree > 2:  # Add reverse combination for degree > 2
                            term = [j] * (degree - 1) + [i]
                            terms.append(term)
        
        return terms
    
    def apply_polynomial_features(self, X):
        """Transform input features into polynomial features"""
        n_samples, n_features = X.shape
        terms = self.generate_polynomial_terms(n_features)
        
        # Initialize the transformed features array
        X_transformed = np.ones((n_samples, 1 + len(terms)))  # +1 for bias term
        
        # Calculate each polynomial term
        for i, combo in enumerate(terms):
            term = np.ones(n_samples)
            for feature_idx in combo:
                term *= X[:, feature_idx]
            X_transformed[:, i + 1] = term
            
        return X_transformed
    
    def select_random_features(self, X, iter):
        n_total_features = X.shape[1]
        
        if self.polynomial_degree > 1:
            # Calculate number of features to select based on percentage
            n_features_to_select = max(1, int(self.num_nonlinear_features * n_total_features))
            n_features_to_select = min(n_features_to_select, n_total_features)
            
            if self.random_state is not None:
                np.random.seed(self.random_state + iter)
            
            # Always include the bias term (index 0) and randomly select other features
            selected_indices = np.concatenate([
                [0],  # bias term
                1 + np.random.choice(n_total_features - 1, n_features_to_select - 1, replace=False)
            ])
            
            return np.sort(selected_indices)
        
        return None
    
    def calculate_loss(self, X, y, weights):
        pred = self.sigmoid(X @ weights.T)
        loss = np.sum(((y-pred) * pred * (1-pred)).reshape(-1,1) * X, axis=0)
        return loss
    
    def calculate_error(self, X, y, weights):
        pred = self.sigmoid(X @ weights.T)
        return np.sum((y - pred) ** 2)
    
    def fit_single_model(self, X, y, X_val=None, y_val=None, curr_iteration=-1):
        num_features = X.shape[1]
        weights = self.init_weights(num_features, curr_iteration)
        patience_counter = 0
        best_weights = None
        best_loss = float('inf')
        min_iter = 10
        
        for iter in range(self.iterations):
            loss = self.calculate_loss(X, y, weights)
            weights = weights + self.learning_rate * loss
            
            if X_val is not None and y_val is not None:
                val_error = self.calculate_error(X_val, y_val, weights)
                
                if val_error < best_loss:
                    best_loss = val_error
                    best_weights = weights.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience and iter > min_iter:
                    break

            if np.abs(loss).max() < 1e-6 and iter > min_iter:
                break

        return best_weights, best_loss
    
    def fit_single_model_wrapper(self, X, y, iter):
        n_samples, n_features = X.shape
        bag_samples = int(self.bag_size * n_samples)
        feature_subset = int(self.num_features * n_features) if self.num_features else n_features
        
        if self.random_state is not None:
            np.random.seed(self.random_state + iter)
        
        # Bagging: random sample with replacement
        idx = np.random.choice(n_samples, bag_samples, replace=True)
        
        # Create the mask for the bagged instances
        bag_mask = np.zeros(n_samples, dtype=bool)
        bag_mask[idx] = True
        
        # Split the data to create out of bagged instances
        X_bag, y_bag = X[idx], y[idx]
        X_val, y_val = X[~bag_mask], y[~bag_mask]
        
        # Random feature subset
        feature_idx = np.random.choice(n_features, feature_subset, replace=False)
        X_subset = X_bag[:, feature_idx]
        
        # Transform features and select nonlinear terms
        X_transformed = self.apply_polynomial_features(X_subset)
        nonlinear_feature_idx = self.select_random_features(X_transformed, iter)
        
        if nonlinear_feature_idx is not None:
            X_transformed = X_transformed[:, nonlinear_feature_idx]
            if X_val is not None:
                X_val_transformed = self.apply_polynomial_features(X_val[:, feature_idx])
                X_val_transformed = X_val_transformed[:, nonlinear_feature_idx]
        else:
            if X_val is not None:
                X_val_transformed = self.apply_polynomial_features(X_val[:, feature_idx])

        if X_val is not None:
            # Fit model
            weights, best_loss = self.fit_single_model(X_transformed, y_bag, X_val_transformed, y_val)
        else:
            # Fit model
            weights, best_loss = self.fit_single_model(X_transformed, y_bag)
        
        return (feature_idx, weights), best_loss, nonlinear_feature_idx
    
    def fit(self, X, y):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single_model_wrapper)(X, y, i) 
            for i in range(self.num_models)
        )
        self.models, self.validation_error, self.feature_nonlinear_index = zip(*results)
    
    def predict_proba(self, X):
        probas = []
        for (feature_idx, weights), nonlinear_idx in zip(self.models, self.feature_nonlinear_index):
            X_subset = X[:, feature_idx]
            X_transformed = self.apply_polynomial_features(X_subset)
            if nonlinear_idx is not None:
                X_transformed = X_transformed[:, nonlinear_idx]
            probas.append(self.sigmoid(X_transformed @ weights))
        return np.mean(probas, axis=0)
            
    def predict(self, X):
        return [0 if u < 0.5 else 1 for u in self.predict_proba(X)]
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def f1_score(self, y, y_pred):
        TP = np.sum((y == 1) & (y_pred == 1))
        FP = np.sum((y == 0) & (y_pred == 1))
        FN = np.sum((y == 1) & (y_pred == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1