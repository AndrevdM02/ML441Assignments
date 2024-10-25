import numpy as np #type: ignore

class SCG:
    def __init__(self, sigma=1e-4, lambda_parameter=1e-6, num_hidden=5, output_activation='softmax', bias=-1, random_state=None, epochs=1000, reg_parameter=0, debug=False, num_batches=1):
        self.sigma = sigma
        self.lambda_parameter = lambda_parameter
        self.lambda_bar = 0
        self.num_classes = 0
        self.random_state = random_state
        self.num_hidden = num_hidden
        self.output_activation = output_activation
        self.weights = None
        self.bias = bias
        self.num_input = None
        self.num_output = None
        self.epochs = epochs
        self.validation_error = []
        self.training_error = []
        self.reg_parameter = reg_parameter
        self.debug = debug
        self.num_batches = num_batches

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def initialize_weights(self, num_input, num_output):
        np.random.seed(self.random_state)
        
        self.num_input = num_input
        self.num_output = num_output
        
        hidden_bounds = 1 / np.sqrt(num_input)
        output_bounds = 1 / np.sqrt(self.num_hidden)
        
        hidden_weights = np.random.uniform(-hidden_bounds, hidden_bounds, (self.num_hidden, num_input + 1))
        output_weights = np.random.uniform(-output_bounds, output_bounds, (num_output, self.num_hidden + 1))
        
        self.weights = np.concatenate([hidden_weights.flatten(), output_weights.flatten()])

    def get_hidden_weights(self, weights):
        hidden_size = self.num_hidden * (self.num_input + 1)
        return weights[:hidden_size].reshape(self.num_hidden, self.num_input + 1)

    def get_output_weights(self, weights):
        hidden_size = self.num_hidden * (self.num_input + 1)
        return weights[hidden_size:].reshape(self.num_output, self.num_hidden + 1)

    def forward_pass(self, X, weights):
        num_samples = X.shape[0]
        X_bias = self.bias * np.ones(num_samples)
        X_with_bias = np.column_stack([X, X_bias])
        
        hidden_weights = self.get_hidden_weights(weights)
        output_weights = self.get_output_weights(weights)
        
        hidden_output = self.relu(np.dot(X_with_bias, hidden_weights.T))
        hidden_bias = self.bias * np.ones(num_samples)
        hidden_output_with_bias = np.column_stack([hidden_output, hidden_bias])
        
        if self.output_activation == 'sigmoid':
            output = self.sigmoid(np.dot(hidden_output_with_bias, output_weights.T))
        elif self.output_activation == 'softmax':
            output = self.softmax(np.dot(hidden_output_with_bias, output_weights.T))
        elif self.output_activation == 'linear':
            output = self.linear(np.dot(hidden_output_with_bias, output_weights.T))
        else:
            raise ValueError("Invalid output activation function")
        
        return hidden_output, output

    def calculate_error(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if self.output_activation == 'softmax':
            error = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        elif self.output_activation == 'linear':
            error = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
        else:
            error = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / y_true.shape[0]
        
        error += 0.5 * self.reg_parameter * (np.sum(self.weights**2))
        return error

    def calculate_error_for_weights(self, weights, X, y):
        _, output = self.forward_pass(X, weights)
        return self.calculate_error(y, output)

    def calculate_gradient(self, weights, X, y):
        num_samples = X.shape[0]
        X_bias = self.bias * np.ones(num_samples)
        X_with_bias = np.column_stack([X, X_bias])
        
        hidden_output, output = self.forward_pass(X, weights)
        hidden_bias = self.bias * np.ones(num_samples)
        hidden_output_with_bias = np.column_stack([hidden_output, hidden_bias])
        
        if self.output_activation == 'softmax':
            delta_output = output - y
        elif self.output_activation == 'linear':
            delta_output = 2 * (output - y) / num_samples  # Derivative of MSE
        else:
            delta_output = (output - y) * self.sigmoid_derivative(output)
        
        output_weights = self.get_output_weights(weights)
        delta_hidden = np.dot(delta_output, output_weights[:, :-1]) * self.relu_derivative(hidden_output)
        
        grad_output = np.dot(delta_output.T, hidden_output_with_bias) / num_samples
        grad_hidden = np.dot(delta_hidden.T, X_with_bias) / num_samples
        
        total_gradient = np.concatenate([grad_hidden.flatten(), grad_output.flatten()])

        # Add the gradient of the L2 regularization term
        total_gradient += self.reg_parameter * weights

        return total_gradient

    def fit(self, X, y, X_val=None, y_val=None):
        self.validation_error = []
        self.training_error = []
        self.lambda_bar = 0
        self.lambda_parameter = 1e-6

        self.num_classes = y.shape[1]
        self.initialize_weights(X.shape[1], self.num_classes)
        best_weights = None
        best_val_error = float('inf')
        prev_training_error = float('inf')
        prev_weights = np.zeros(len(self.weights))
        small_weights = False
        patience = 20
        
        # Set p=r=-E'(w_1), k=1 and success=True
        success = True
        k = 1
        p = r = -self.calculate_gradient(self.weights, X, y)
                
        while k <= self.epochs:
            # Create batches
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            batches = np.array_split(indices, self.num_batches)
            
            for batch_indices in batches:
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                # Step 2: if success = True calculate the second-order information
                if success:
                    # Calculate σ(k)
                    current_sigma = self.sigma / np.linalg.norm(p)
                    # Calculate s(k)
                    modified_weights = self.weights + (current_sigma * p)
                    s = (self.calculate_gradient(modified_weights, X_batch, y_batch) - self.calculate_gradient(self.weights, X_batch, y_batch))
                    s = s/current_sigma
                    # Calculate δ(k)
                    delta_k = p.T @ s
                
                # Step 3: Scale s(k) and δ(k)
                s += (self.lambda_parameter - self.lambda_bar) * p
                delta_k += (self.lambda_parameter - self.lambda_bar) * np.linalg.norm(p)**2
                
                # Step 4: If δ(k) <= 0 then make the Hessian matrix prositive definite
                if delta_k <= 0:
                    # s(k)
                    s = s + (self.lambda_parameter-(2*delta_k/np.linalg.norm(p)**2))*p
                    # ¯λ(k)
                    self.lambda_bar = 2 * (self.lambda_parameter - (2*delta_k/np.linalg.norm(p)**2))
                    # δ(k)
                    delta_k = -delta_k + (self.lambda_parameter*np.linalg.norm(p)**2)
                    # λ(k)
                    self.lambda_parameter = self.lambda_bar

                # Step 5: Calculate the step size
                # μ(k)
                mu_k = p.T @ r
                # η(k)
                eta_k = mu_k / delta_k
                
                # Step 6: Calculate the comparison parameter
                new_weights = self.weights + (eta_k * p)
                comparison = (2 * delta_k * (self.calculate_error_for_weights(self.weights, X_batch, y_batch) - self.calculate_error_for_weights(new_weights, X_batch, y_batch))) / (mu_k**2)

                # Step 7: If comparison >= 0 then a successful reduction in error can be made
                if comparison >= 0:
                    prev_weights = self.weights
                    self.weights = self.weights + (eta_k * p)
                    
                    if np.abs(np.linalg.norm(self.weights) - np.linalg.norm(prev_weights)) < 1e-6:
                        small_weights = True
                    else:
                        small_weights = False

                    prev_r = r
                    r = -self.calculate_gradient(self.weights, X_batch, y_batch)
                    self.lambda_bar = 0
                    success = True

                    if k % len(self.weights) == 0:
                        p = r
                        break

                    else:
                        Beta_k = ((np.linalg.norm(r)**2) - (r.T @ prev_r)) / mu_k
                        p = r + (Beta_k * p)
                    
                    if comparison >= 0.75:
                        self.lambda_parameter = self.lambda_parameter / 2
                
                else:
                    self.lambda_bar = self.lambda_parameter
                    success = False
                
                # Step 8: If comparison < 0.25, then increase the scale parameter
                if comparison < 0.25:
                    self.lambda_parameter = self.lambda_parameter * 4

            # Calculate the training and validation error and check for overfitting
            t_error = self.calculate_error_for_weights(self.weights, X, y)
            val_error = self.calculate_error_for_weights(self.weights,X_val,y_val)
            if self.debug:
                print(f"Training error is: {t_error} and validation error is: {val_error}")
            self.training_error.append(t_error)
            self.validation_error.append(val_error)

            if val_error < best_val_error:
                best_val_error = val_error
                best_weights = self.weights

            if val_error > np.mean(self.validation_error[min(5,len(self.validation_error)):]) + np.std(self.validation_error[min(5,len(self.validation_error)):]) and prev_training_error > t_error: 
                if self.debug:
                    print(f"Early stopping due to increasing validation error.")
                self.weights = best_weights
                break
            prev_training_error = t_error

            if small_weights and k > patience:
                self.weights = best_weights
                if self.debug:
                    print(f"Early stopping due to small weight changes.")
                break

            if k == self.epochs:
                self.weights = best_weights

            # Step 9: If the steepest descent direction r not equal to a zero vector,
            # then set k = k + 1 and go to 2 else terminate and return w, as the desired minimum.
            if np.allclose(r, 0, atol=1e-6):
                break
            else:
                k += 1

    def score(self, X, y):
        y_pred = self.predict(X)

        if self.output_activation == 'sigmoid':
            y_pred = [0 if u < 0.5 else 1 for u in y_pred]
            score = (np.asarray(y_pred) == y.flatten()).sum() / len(y)
        elif self.output_activation == 'softmax':
            score = np.sum(y_pred.argmax(axis=1) == y.argmax(axis=1)) / len(y)
        else:
            score = np.mean((y - y_pred)**2)
        
        return score

    def predict(self, X):
        _, output = self.forward_pass(X, self.weights)
        return output