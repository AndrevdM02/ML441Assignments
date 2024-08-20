# import numpy as np #type: ignore
# import pandas as pd #type: ignore
# import math
# from collections import Counter
# from sklearn.model_selection import train_test_split #type: ignore
# from joblib import Parallel, delayed #type: ignore
# import random

# class Node:
#     def __init__(self, is_leaf=False, branches=None, prediction=None, test=None):
#         self.is_leaf = is_leaf
#         self.branches = branches or {}
#         self.prediction = prediction
#         self.test = test

# class ClassificationTree:
#     def __init__(self, max_depth=None, min_samples_split=2, random_state=None, data_split_size=0.2, class_weights=None):
#         self.root = None
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.random_state = random_state
#         self.data_split_size = data_split_size
#         self.class_weights = class_weights
#         self.feature_names = None  # Initialize as None
#         self.class_to_index = None

#     def fit(self, X, y):

#         if self.class_weights is None:
#             random.seed(self.random_state)
#             X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.data_split_size, random_state=self.random_state)
#             y_train = pd.Series(y_train)
#             y_val = pd.Series(y_val)
#             # Compute class weights if not provided
#             class_counts = y_train.value_counts()
#             total_samples = len(y_train)
#             class_labels = class_counts.index
#             self.class_weights = np.array([total_samples / class_counts[cls] for cls in class_labels])
#             self.class_to_index = {cls: idx for idx, cls in enumerate(class_labels)}
#         else:
#             self.class_to_index = {cls: idx for idx, cls in enumerate(np.unique(y_train))}

#         self.feature_names = X.columns.tolist()  # Set feature names here

#         self.root = self.induce_tree(X_train, y_train, default_class=self.majority_class(y_train), depth=0)
#         self.prune_tree(X_val, y_val)
#         return self


#     def induce_tree(self, X, y, default_class=None, depth=0):
#         X = X.reset_index(drop=True)
#         y = y.reset_index(drop=True)

#         indent = "   " * depth
#         print(f"{indent}Inducing tree for {len(X)} samples")

#         if len(X) == 0 or len(y) == 0:
#             return Node(is_leaf=True, prediction=default_class)

#         if len(X) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
#             return Node(is_leaf=True, prediction=self.majority_class(y))
        
#         if y.nunique() == 1:
#             return Node(is_leaf=True, prediction=y.iloc[0])
        
#         best_feature = self.select_best_feature(X, y)
#         print(f"Best feature to split on: {best_feature}")

#         if best_feature is None:
#             return Node(is_leaf=True, prediction=self.majority_class(y))
        
#         subset = self.split_dataset(X, y, best_feature)
#         y_class = self.majority_class(y)

#         branches_dict = {}
#         for value, (X_subset, y_subset) in subset.items():
#             branches_dict[value] = self.induce_tree(X_subset, y_subset, depth=depth+1, default_class=y_class)

#         return Node(is_leaf=False, test=best_feature, branches=branches_dict)

#     def entropy(self, y):
#         counts = y.value_counts()
#         epsilon = 1e-5
#         weighted_counts = np.array([self.class_weights[self.class_to_index[cls]] * count for cls, count in counts.items()])
#         probs = weighted_counts / (np.sum(weighted_counts) + epsilon)
#         return -np.sum(probs * np.log2(probs + epsilon))

#     def information_gain(self, X, y, feature):
#         total_entropy = self.entropy(y)

#         feature_values = X[feature]
#         unique_values = feature_values.dropna().unique()

#         weighted_entropy = 0.0
#         missing_value_fraction = feature_values.isna().mean()

#         for value in unique_values:
#             if pd.isna(value):
#                 continue

#             subset_y = y[feature_values == value]
#             prob = len(subset_y) / len(y)
#             if len(subset_y) > 0:
#                 weighted_entropy += prob * self.entropy(subset_y)

#         known_data_entropy = total_entropy - weighted_entropy
#         weighted_entropy = (1 - missing_value_fraction) * known_data_entropy

#         return total_entropy - weighted_entropy

#     def split_info(self, X, y, feature):
#         feature_values = X[feature]
#         subset_size = y.groupby(feature_values).size()
#         probs = subset_size / len(y)

#         split_info_value = -np.sum(probs * np.log2(probs + 1e-5))

#         missing_value_fraction = feature_values.isna().mean()

#         if missing_value_fraction > 0:
#             split_info_value -= missing_value_fraction * math.log2(missing_value_fraction + 1e-5)

#         return split_info_value
    
#     def gain_ratio(self, X, y, feature):
#         gain = self.information_gain(X, y, feature)
#         split_info_value = self.split_info(X, y, feature)

#         return gain / split_info_value if split_info_value != 0 else 0

#     def select_best_feature(self, X, y):
#         n_jobs = -1

#         results = Parallel(n_jobs=n_jobs)(
#             delayed(self.compute_feature_score)(X,y,feature) for feature in X.columns
#         )

#         results = [result for result in results if result is not None]

#         if not results:
#             return None

#         best_feature, best_gain_ratio = max(results, key=lambda x: x[1])
        
#         return best_feature

#     def compute_feature_score(self, X, y, feature):
#         if X[feature].dtype in ['object', 'category']:
#             ratio = self.gain_ratio(X, y, feature)
#             return feature, ratio if ratio > 0 else 0
#         else:
#             ratio, _ = self.best_split_numerical_feature(X, y, feature)
#             return feature, ratio if ratio > 0 else 0

#     def best_split_numerical_feature(self, X, y, feature):
#         feature_values = X[feature].values
#         sorted_indices = np.argsort(feature_values)
#         sorted_values = feature_values[sorted_indices]
#         sorted_labels = y.values[sorted_indices]

#         unique_values, counts = np.unique(sorted_values, return_counts=True)
#         num_unique = len(unique_values)

#         if num_unique <= 1:
#             return 0, None

#         total_entropy = self.entropy(y)
#         total_samples = len(y)

#         left_count = 0
#         right_count = total_samples
#         left_entropy = 0
#         right_entropy = total_entropy

#         best_gain = 0
#         best_threshold = None

#         left_counts = np.zeros(num_unique)
#         right_counts = counts.copy()

#         for i in range(num_unique - 1):
#             value = unique_values[i]
#             left_count += counts[i]
#             right_count -= counts[i]

#             left_counts[i] = left_count
#             right_counts[i] = right_count

#             left_entropy = self.entropy(sorted_labels[:left_count])
#             right_entropy = self.entropy(sorted_labels[left_count:])
            
#             gain = total_entropy - (left_count / total_samples * left_entropy +
#                                     right_count / total_samples * right_entropy)

#             if gain > best_gain:
#                 best_gain = gain
#                 best_threshold = (value + unique_values[i + 1]) / 2

#         return best_gain, best_threshold if best_gain > 0 else (0,None)

#     def split_dataset(self, X, y, feature):
#         feature_values = X[feature]
        
#         if X[feature].dtype in ['object', 'category']:
#             subset = {}
#             unique_values = feature_values.unique()
#             for value in unique_values:
#                 mask = feature_values == value
#                 X_subset = X[mask].drop(columns=feature)
#                 y_subset = y[mask]
#                 subset[value] = (X_subset, y_subset)
#         else:
#             best_gain, threshold = self.best_split_numerical_feature(X, y, feature)
#             if threshold is None:
#                 return {}

#             subset = {}
#             mask_left = feature_values < threshold
#             mask_right = feature_values >= threshold
#             X_left = X[mask_left].drop(columns=feature)
#             y_left = y[mask_left]
#             X_right = X[mask_right].drop(columns=feature)
#             y_right = y[mask_right]
            
#             subset[f"< {threshold}"] = (X_left, y_left)
#             subset[f">= {threshold}"] = (X_right, y_right)
        
#         return subset

#     def majority_class(self, y):
#         if y.empty:
#             raise ValueError("Cannot compute majority class for an empty series.")
        
#         most_frequent = y.mode()

#         if len(most_frequent) > 1:
#             return random.choice(y.dropna().unique().tolist())
#         return most_frequent[0]
    
#     def print_tree(self, node, depth=0):
#         indent = "  " * depth

#         if node.is_leaf:
#             print(f"{indent}Leaf: Predict {node.prediction}")
#         else:
#             print(f"{indent}{depth} Node: Split on '{node.test}'")
#             for value, branch in node.branches.items():
#                 print(f"{indent}  If {node.test} == {value}:")
#                 self.print_tree(branch, depth + 1)
    
#     def predict_single(self, node, instance):
#         """
#         Predict the class label for a single instance using the provided decision tree node.

#         Parameters:
#         - node: The current node in the decision tree.
#         - instance: A named tuple containing the feature values of the instance to be predicted.

#         Returns:
#         - The predicted class label.
#         """
#         if node.is_leaf:
#             return node.prediction

#         # Access the feature value using dot notation for named tuples
#         feature_value = getattr(instance, node.test)

#         if isinstance(feature_value, str):  # Handling categorical features
#             if feature_value in node.branches:
#                 return self.predict_single(node.branches[feature_value], instance)
#             else:
#                 # If the feature value is not found in branches, return the majority class
#                 return node.prediction

#         # Handling numerical features
#         else:
#             for branch_value, branch_node in node.branches.items():
#                 if isinstance(branch_value, str):
#                     if '>= ' in branch_value:
#                         threshold = float(branch_value.replace('>= ', ''))
#                         if feature_value >= threshold:
#                             return self.predict_single(branch_node, instance)
#                     elif '< ' in branch_value:
#                         threshold = float(branch_value.replace('< ', ''))
#                         if feature_value < threshold:
#                             return self.predict_single(branch_node, instance)
#                 else:
#                     # For numerical branch values
#                     if feature_value >= branch_value:
#                         return self.predict_single(branch_node, instance)

#         # Fallback in case none of the conditions matched
#         return node.prediction
    
#     def predict(self, X):
#         """
#         Predict class labels for a DataFrame using the trained decision tree.

#         Parameters:
#         - X: A pandas DataFrame with feature values.

#         Returns:
#         - A pandas Series with predicted class labels.
#         """
#         # Ensure that X is a DataFrame
#         if not isinstance(X, pd.DataFrame):
#             raise ValueError("X must be a pandas DataFrame")

#         # Apply the predict_single method to each row of the DataFrame
#         predictions = X.apply(lambda row: self.predict_single(self.root, row), axis=1)

#         return predictions
    
#     def prune_tree(self, X_val, y_val):
#         """
#         Perform post-pruning on the decision tree using a validation set.
        
#         Parameters:
#         - X_val: Validation set features (pandas DataFrame)
#         - y_val: Validation set target (pandas Series)
#         """
#         if self.root is None:
#             raise ValueError("The tree has not been trained yet.")

#         self._prune_node(self.root, X_val, y_val)

#     def _prune_node(self, node, X_val, y_val):
#         """
#         Recursively prune the tree starting from the given node.
        
#         Parameters:
#         - node: The current node in the decision tree.
#         - X_val: Validation set features (pandas DataFrame)
#         - y_val: Validation set target (pandas Series)
#         """
#         # If the node is a leaf, no need to prune
#         if node.is_leaf:
#             return

#         # Recursively prune child nodes first
#         for value, branch in node.branches.items():
#             # Filter validation set based on the branch value
#             if isinstance(value, str) and ('>= ' in value or '< ' in value):  # Handling categorical features
#                 if '>= ' in value:
#                     threshold = float(value.replace('>= ', ''))
#                     mask = X_val[node.test] >= threshold
#                 elif '< ' in value:
#                     threshold = float(value.replace('< ', ''))
#                     mask = X_val[node.test] < threshold
#             else:
#                 mask = X_val[node.test] == value

#             X_val_subset = X_val[mask]
#             y_val_subset = y_val[mask]
            
#             if not X_val_subset.empty:
#                 self._prune_node(branch, X_val_subset, y_val_subset)

#         if not X_val.empty:
#             # After pruning child nodes, consider pruning this node
#             original_predictions = self.predict(X_val)
#             original_accuracy = np.mean(original_predictions == y_val)

#             # Convert the node to a leaf
#             original_branches = node.branches
#             node.is_leaf = True
#             node.branches = {}
#             node.prediction = self.majority_class(y_val)

#             pruned_predictions = self.predict(X_val)
#             pruned_accuracy = np.mean(pruned_predictions == y_val)

#             # If pruning doesn't improve accuracy, revert the node back to its original state
#             if pruned_accuracy < original_accuracy:
#                 node.is_leaf = False
#                 node.branches = original_branches
#                 node.prediction = None

import numpy as np #type: ignore
import pandas as pd #type: ignore
import math
from collections import Counter
from sklearn.model_selection import train_test_split #type: ignore
from joblib import Parallel, delayed #type: ignore
import random

class Node:
    def __init__(self, is_leaf=False, branches=None, prediction=None, test=None):
        self.is_leaf = is_leaf
        self.branches = branches or {}
        self.prediction = prediction
        self.test = test

class ClassificationTree:
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None, data_split_size=0.2, class_weights=None):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.data_split_size = data_split_size
        self.class_weights = class_weights
        self.feature_names = None  # Initialize as None
        self.class_to_index = None
        self.classes_ = None

    def fit(self, X, y):

        if self.class_weights is None:
            random.seed(self.random_state)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.data_split_size, random_state=self.random_state)
            y_train = pd.Series(y_train)
            y_val = pd.Series(y_val)
            # Compute class weights if not provided
            class_counts = y_train.value_counts()
            total_samples = len(y_train)
            class_labels = class_counts.index
            self.class_weights = np.array([total_samples / class_counts[cls] for cls in class_labels])
            self.class_to_index = {cls: idx for idx, cls in enumerate(class_labels)}
        else:
            self.class_to_index = {cls: idx for idx, cls in enumerate(np.unique(y_train))}

        self.feature_names = X.columns.tolist()  # Set feature names here
        self.classes_ = np.unique(np.asarray(y))

        self.root = self.induce_tree(X_train, y_train, default_class=self.majority_class(y_train), depth=0)
        self.prune_tree(X_val, y_val)
        return self


    def induce_tree(self, X, y, default_class, depth=0):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        indent = "   " * depth
        print(f"{indent}Inducing tree for {len(X)} samples")

        if len(X) == 0 or len(y) == 0:
            return Node(is_leaf=True, prediction=default_class)

        if len(X) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return Node(is_leaf=True, prediction=self.majority_class(y))
        
        if y.nunique() == 1:
            return Node(is_leaf=True, prediction=self.majority_class(y))
        
        best_feature = self.select_best_feature(X, y)
        print(f"Best feature to split on: {best_feature}")

        if best_feature is None:
            return Node(is_leaf=True, prediction=self.majority_class(y))
        
        subset = self.split_dataset(X, y, best_feature)
        y_class = self.majority_class(y)

        branches_dict = {}
        for value, (X_subset, y_subset) in subset.items():
            branches_dict[value] = self.induce_tree(X_subset, y_subset, depth=depth+1, default_class=y_class)

        return Node(is_leaf=False, test=best_feature, branches=branches_dict)

    def entropy(self, y):
        counts = y.value_counts()
        epsilon = 1e-5
        weighted_counts = np.array([self.class_weights[self.class_to_index[cls]] * count for cls, count in counts.items()])
        probs = weighted_counts / (np.sum(weighted_counts) + epsilon)
        return -np.sum(probs * np.log2(probs + epsilon))

    def information_gain(self, X, y, feature):
        total_entropy = self.entropy(y)

        feature_values = X[feature]
        unique_values = feature_values.dropna().unique()

        weighted_entropy = 0.0
        missing_value_fraction = feature_values.isna().mean()

        for value in unique_values:
            if pd.isna(value):
                continue

            subset_y = y[feature_values == value]
            prob = len(subset_y) / len(y)
            if len(subset_y) > 0:
                weighted_entropy += prob * self.entropy(subset_y)

        known_data_entropy = total_entropy - weighted_entropy
        weighted_entropy = (1 - missing_value_fraction) * known_data_entropy

        return total_entropy - weighted_entropy

    def split_info(self, X, y, feature):
        feature_values = X[feature]
        subset_size = y.groupby(feature_values).size()
        probs = subset_size / len(y)

        split_info_value = -np.sum(probs * np.log2(probs + 1e-5))

        missing_value_fraction = feature_values.isna().mean()

        if missing_value_fraction > 0:
            split_info_value -= missing_value_fraction * math.log2(missing_value_fraction + 1e-5)

        return split_info_value
    
    def gain_ratio(self, X, y, feature):
        gain = self.information_gain(X, y, feature)
        split_info_value = self.split_info(X, y, feature)

        return gain / split_info_value if split_info_value != 0 else 0

    def select_best_feature(self, X, y):
        
        valid_features = [feature for feature in X.columns if not X[feature].isna().any()]

        if not valid_features:
            return None

        n_jobs = -1

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.compute_feature_score)(X,y,feature) for feature in valid_features
        )

        results = [result for result in results if result is not None]

        if not results:
            return None

        best_feature, best_gain_ratio = max(results, key=lambda x: x[1])
        
        return best_feature

    def compute_feature_score(self, X, y, feature):
        if X[feature].dtype in ['object', 'category']:
            ratio = self.gain_ratio(X, y, feature)
            return feature, ratio if ratio > 0 else 0
        else:
            ratio, _ = self.best_split_numerical_feature(X, y, feature)
            return feature, ratio if ratio > 0 else 0

    def best_split_numerical_feature(self, X, y, feature):
        feature_values = X[feature].values
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_labels = y.values[sorted_indices]

        unique_values, counts = np.unique(sorted_values, return_counts=True)
        num_unique = len(unique_values)

        if num_unique <= 1:
            return 0, None

        total_entropy = self.entropy(y)
        total_samples = len(y)

        left_count = 0
        right_count = total_samples
        left_entropy = 0
        right_entropy = total_entropy

        best_gain = 0
        best_threshold = None

        left_counts = np.zeros(num_unique)
        right_counts = counts.copy()

        for i in range(num_unique - 1):
            value = unique_values[i]
            left_count += counts[i]
            right_count -= counts[i]

            left_counts[i] = left_count
            right_counts[i] = right_count

            left_entropy = self.entropy(sorted_labels[:left_count])
            right_entropy = self.entropy(sorted_labels[left_count:])
            
            gain = total_entropy - (left_count / total_samples * left_entropy +
                                    right_count / total_samples * right_entropy)

            if gain > best_gain:
                best_gain = gain
                best_threshold = (value + unique_values[i + 1]) / 2

        return best_gain, best_threshold if best_gain > 0 else (0,None)

    def split_dataset(self, X, y, feature):
        feature_values = X[feature]
        
        if X[feature].dtype in ['object', 'category']:
            subset = {}
            unique_values = feature_values.dropna().unique()
            for value in unique_values:
                mask = feature_values == value
                X_subset = X[mask].drop(columns=feature)
                y_subset = y[mask]
                subset[value] = (X_subset, y_subset)
        else:
            best_gain, threshold = self.best_split_numerical_feature(X, y, feature)
            if threshold is None:
                return {}

            subset = {}
            mask_left = feature_values < threshold
            mask_right = feature_values >= threshold
            X_left = X[mask_left].drop(columns=feature)
            y_left = y[mask_left]
            X_right = X[mask_right].drop(columns=feature)
            y_right = y[mask_right]
            
            subset[f"< {threshold}"] = (X_left, y_left)
            subset[f">= {threshold}"] = (X_right, y_right)
        
        return subset

    def majority_class(self, y):
        if y.empty:
            raise ValueError("Cannot compute majority class for an empty series.")
        
        most_frequent = y.mode()

        if len(most_frequent) > 1:
            return random.choice(y.dropna().unique().tolist())
        return most_frequent[0]
    
    def print_tree(self, node, depth=0):
        indent = "  " * depth

        if node.is_leaf:
            print(f"{indent}Leaf: Predict {node.prediction}")
        else:
            print(f"{indent}{depth} Node: Split on '{node.test}'")
            for value, branch in node.branches.items():
                print(f"{indent}  If {node.test} == {value}:")
                self.print_tree(branch, depth + 1)
    
    def predict_single(self, node, instance):
        """
        Predict the class label for a single instance using the provided decision tree node.

        Parameters:
        - node: The current node in the decision tree.
        - instance: A named tuple containing the feature values of the instance to be predicted.

        Returns:
        - The predicted class label.
        """
        if node.is_leaf:
            return node.prediction

        # Access the feature value using dot notation for named tuples
        feature_value = getattr(instance, node.test)

        if feature_value is np.nan or pd.isna(feature_value):  # Handle missing feature value
            probabilities = self.calculate_branch_probability(node, None)
            return max(probabilities, key=probabilities.get)

        if isinstance(feature_value, str):  # Handling categorical features
            if feature_value in node.branches:
                return self.predict_single(node.branches[feature_value], instance)
            else:
                # If the feature value is not found in branches, return the majority class
                return random.choice(self.classes_)

        # Handling numerical features
        else:
            for branch_value, branch_node in node.branches.items():
                if isinstance(branch_value, str):
                    if '>= ' in branch_value:
                        threshold = float(branch_value.replace('>= ', ''))
                        if feature_value >= threshold:
                            return self.predict_single(branch_node, instance)
                    elif '< ' in branch_value:
                        threshold = float(branch_value.replace('< ', ''))
                        if feature_value < threshold:
                            return self.predict_single(branch_node, instance)
                else:
                    # For numerical branch values
                    if feature_value >= branch_value:
                        return self.predict_single(branch_node, instance)

        # Fallback in case none of the conditions matched
        return node.prediction
    
    def predict(self, X):
        """
        Predict class labels for a DataFrame using the trained decision tree.

        Parameters:
        - X: A pandas DataFrame with feature values.

        Returns:
        - A pandas Series with predicted class labels.
        """
        # Ensure that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Apply the predict_single method to each row of the DataFrame
        predictions = X.apply(lambda row: self.predict_single(self.root, row), axis=1)

        return predictions
    
    def prune_tree(self, X_val, y_val):
        """
        Perform post-pruning on the decision tree using a validation set.
        
        Parameters:
        - X_val: Validation set features (pandas DataFrame)
        - y_val: Validation set target (pandas Series)
        """
        if self.root is None:
            raise ValueError("The tree has not been trained yet.")

        self._prune_node(self.root, X_val, y_val)

    def _prune_node(self, node, X_val, y_val):
        """
        Recursively prune the tree starting from the given node.
        
        Parameters:
        - node: The current node in the decision tree.
        - X_val: Validation set features (pandas DataFrame)
        - y_val: Validation set target (pandas Series)
        """
        # If the node is a leaf, no need to prune
        if node.is_leaf:
            return

        # Recursively prune child nodes first
        for value, branch in node.branches.items():
            # Filter validation set based on the branch value
            if isinstance(value, str) and ('>= ' in value or '< ' in value):  # Handling categorical features
                if '>= ' in value:
                    threshold = float(value.replace('>= ', ''))
                    mask = X_val[node.test] >= threshold
                elif '< ' in value:
                    threshold = float(value.replace('< ', ''))
                    mask = X_val[node.test] < threshold
            else:
                mask = X_val[node.test] == value

            X_val_subset = X_val[mask]
            y_val_subset = y_val[mask]
            
            if not X_val_subset.empty:
                self._prune_node(branch, X_val_subset, y_val_subset)

        if not X_val.empty:
            # After pruning child nodes, consider pruning this node
            original_predictions = self.predict(X_val)
            original_accuracy = np.mean(original_predictions == y_val)

            # Convert the node to a leaf
            prev_prediction = node.prediction
            original_branches = node.branches
            node.is_leaf = True
            node.branches = {}
            node.prediction = self.majority_class(y_val)

            pruned_predictions = self.predict(X_val)
            pruned_accuracy = np.mean(pruned_predictions == y_val)

            # If pruning doesn't improve accuracy, revert the node back to its original state
            if pruned_accuracy < original_accuracy:
                node.is_leaf = False
                node.branches = original_branches
                node.prediction = prev_prediction

    def calculate_branch_probability(self, branch_node, parent_node):
        """
        Calculate the probability of predicting each class based on the branch node.

        Parameters:
        - branch_node: The current branch node of the decision tree.
        - parent_node: The parent node from which this branch node was reached.

        Returns:
        - A dictionary where the keys are class labels and the values are their probabilities.
        """
        if branch_node.is_leaf:
            # If the branch node is a leaf, return the probability of the predicted class at this leaf
            class_counts = Counter({branch_node.prediction: 1})
            total_counts = 1
        else:
            # If the branch node is not a leaf, compute the class probabilities for this branch node
            class_counts = Counter()
            for value, sub_branch in branch_node.branches.items():
                # Recursively compute the probabilities for each branch
                sub_class_counts = self.calculate_branch_probability(sub_branch, branch_node)
                for cls, count in sub_class_counts.items():
                    class_counts[cls] += count

            total_counts = sum(class_counts.values())

        # Compute probabilities based on counts
        probabilities = {cls: count / total_counts for cls, count in class_counts.items()}
        return probabilities