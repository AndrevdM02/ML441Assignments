import numpy as np #type: ignore

class SGD:
    prev_hidden_weights = 0
    prev_output_weights = 0

    def __init__(self, eta=0.1, alpha=0.1, epochs=5, random_state=None, num_hidden=10, bias=-1, output_activation='softmax', min_avg_weight_change=1e-6, reg_parameter = 0, debug=False):
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.random_state = random_state
        self.num_hidden = num_hidden
        self.num_classes = None
        self.bias = bias
        self.hidden_weights = None
        self.out_weights = None
        self.output_activation = output_activation
        self.validation_error = []
        self.training_error = []
        self.min_avg_weight_change = min_avg_weight_change
        self.reg_parameter = reg_parameter
        self.debug = debug

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def initialize_weights(self, num_input, num_output):
        np.random.seed(self.random_state)
        
        hidden_bounds = 1 / np.sqrt(num_input + 1)
        self.hidden_weights = np.random.uniform(-hidden_bounds, hidden_bounds, (self.num_hidden, num_input + 1))
        self.prev_hidden_weights = np.zeros((self.num_hidden, num_input + 1))

        out_bounds = 1 / np.sqrt(self.num_hidden + 1)
        self.out_weights = np.random.uniform(-out_bounds, out_bounds, (num_output, self.num_hidden + 1))
        self.prev_output_weights = np.zeros((num_output, self.num_hidden + 1))


    def forward_pass(self, x):
        x_with_bias = np.append(x, self.bias)

        hidden_layer_input = np.dot(self.hidden_weights, x_with_bias)
        hidden_layer_output = self.relu(hidden_layer_input)
        
        hidden_with_bias = np.append(hidden_layer_output, self.bias)
        
        output_layer_input = np.dot(self.out_weights, hidden_with_bias)
        
        if self.output_activation == 'sigmoid':
            output = self.sigmoid(output_layer_input)
        elif self.output_activation == 'softmax':
            output = self.softmax(output_layer_input.reshape(1, -1)).flatten()
        elif self.output_activation == 'linear':
            output = self.linear(output_layer_input)
        else:
            raise ValueError("Invalid output activation function")
        
        return hidden_layer_output, output

    def cross_entropy_error(self, y_true, y_pred):
        epsilon = 1e-15  # Small value to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)

        if self.output_activation == 'softmax':
            ce = -np.sum(y_true * np.log(y_pred))
        elif self.output_activation == 'linear':
            ce = np.mean((y_true - y_pred)**2)  # Mean Squared Error for linear activation
        else:
            ce = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # Add L2/ weight decay regularization term
        ce += 0.5 * self.reg_parameter * (np.sum(self.out_weights**2) + np.sum(self.hidden_weights**2))
        return ce

    def calculate_error_signals(self, y, hidden_output, final_output):    
        # Error signal for output layer (δ_o_k)
        if self.output_activation == 'softmax':
            delta_o = -(y - final_output)  # This is the derivative of cross-entropy with softmax
        elif self.output_activation == 'linear':
            delta_o = -(y - final_output) * self.linear_derivative(final_output)
        else:
            delta_o = -(y - final_output) * self.sigmoid_derivative(final_output)

        # Error signal for hidden layer (δ_y_j)
        delta_y = np.dot(self.out_weights[:, :-1].T, delta_o) * self.relu_derivative(hidden_output)

        return delta_o, delta_y

    def update_weights(self, x, hidden_output, delta_o, delta_y):
        x_with_bias = np.append(x, self.bias)
        hidden_with_bias = np.append(hidden_output, self.bias)

        delta_w_kj = -self.eta * (np.outer(delta_o, hidden_with_bias) + self.reg_parameter * self.out_weights)
        output_weight_change = delta_w_kj + self.alpha * self.prev_output_weights
        self.out_weights += output_weight_change
        self.prev_output_weights = delta_w_kj

        delta_v_ji = -self.eta * (np.outer(delta_y, x_with_bias) + self.reg_parameter * self.hidden_weights)
        hidden_weight_change = delta_v_ji + self.alpha * self.prev_hidden_weights
        self.hidden_weights += hidden_weight_change
        self.prev_hidden_weights = delta_v_ji

        return hidden_weight_change, output_weight_change

    def fit(self, X, y, X_val, y_val):
        self.validation_error = []
        self.training_error = []
        self.num_classes = y.shape[1]
        num_output = self.num_classes
        num_input = X.shape[1]
        
        self.initialize_weights(num_input, num_output)
        t = 0
        best_val_error = float('inf')
        best_epoch = 0
        best_hidden_weights = None
        best_output_weights = None
        prev_training_error = float('inf')
        prev_hidden_weights = self.hidden_weights.copy()
        prev_output_weights = self.out_weights.copy()
        
        while t < self.epochs:
            training_err = 0
            total_weight_change = 0

            # # Randomly shuffle the indices
            rng = np.random.default_rng()
            indices = rng.permutation(X.shape[0])
            for p in indices:
                hidden_output, output = self.forward_pass(X[p])
                delta_o, delta_y = self.calculate_error_signals(y[p], hidden_output, output)
                
                # Update weights and track changes
                hidden_weight_change, output_weight_change = self.update_weights(X[p], hidden_output, delta_o, delta_y)
                total_weight_change += (np.sum(np.abs(hidden_weight_change)) + np.sum(np.abs(output_weight_change))) / (self.hidden_weights.size + self.out_weights.size)

                # Calculate cross-entropy error
                error = self.cross_entropy_error(y[p], output)
                training_err += error

                prev_hidden_weights = self.hidden_weights.copy()
                prev_output_weights = self.out_weights.copy()

            avg_weight_change = total_weight_change / len(indices)
            if self.debug:
                print(f"Epoch {t+1}/{self.epochs}, Error: {training_err/len(y)}, Avg Weight Change: {avg_weight_change}")

            # Optionally, you can calculate and print validation error here
            val_error = self.calculate_error(X_val, y_val)
            if self.debug:
                print(f"Validation Error: {val_error}")

            self.validation_error.append(val_error)
            self.training_error.append(training_err/len(y))
            if val_error < best_val_error:
                best_val_error = val_error
                best_epoch = t
                best_hidden_weights = self.hidden_weights.copy()
                best_output_weights = self.out_weights.copy()

            # Calculate moving average and standard deviation of validation errors
            avg_val_error = np.mean(self.validation_error[min(5,len(self.validation_error)):])
            std_val_error = np.std(self.validation_error[min(5,len(self.validation_error)):])

            # Check for overfitting or small weight changes
            if val_error > avg_val_error + std_val_error or avg_weight_change < self.min_avg_weight_change:
                if prev_training_error > training_err/len(y) and val_error > avg_val_error + std_val_error:
                    stop_reason = "overfitting detected"
                    if self.debug:
                        print(f"Early stopping at epoch {t+1} due to {stop_reason}. Best epoch was {best_epoch+1}")
                    self.hidden_weights = best_hidden_weights
                    self.out_weights = best_output_weights
                    break
                
                if avg_weight_change < self.min_avg_weight_change:
                    stop_reason = "small weight changes"
                    self.hidden_weights = best_hidden_weights
                    self.out_weights = best_output_weights
                    if self.debug:
                        print(f"Early stopping at epoch {t+1} due to {stop_reason}. Best epoch was {best_epoch+1}")
                    break

                if (t+1) == self.epochs:
                    self.hidden_weights = best_hidden_weights
                    self.out_weights = best_output_weights
                    break
            t += 1
            prev_training_error = training_err/len(y)

    def calculate_error(self, X, y):
        total_error = 0
        for p in range(X.shape[0]):
            _, output = self.forward_pass(X[p])
            total_error += self.cross_entropy_error(y[p], output)
        return total_error / X.shape[0]
    
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
        predictions = []
        for x in X:
            _, output = self.forward_pass(x)
            predictions.append(output)
        return np.array(predictions)