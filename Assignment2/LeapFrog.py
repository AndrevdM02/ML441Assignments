import numpy as np #type: ignore

class LFrog:
    def __init__(self, num_hidden=5, epochs=1000, output_activation='softmax', bias=-1, random_state=None, reg_parameter=0, num_batches = 1, debug=False):
        self.num_hidden = num_hidden
        self.epochs = epochs
        self.output_activation = output_activation
        self.bias = bias
        self.random_state = random_state
        self.num_classes = 0
        self.weights = None
        self.num_input = None
        self.num_output = None
        self.validation_error = []
        self.training_error = []
        self.reg_parameter = reg_parameter
        self.num_batches = num_batches
        self.debug = debug

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
        self.num_classes = y.shape[1]
        self.initialize_weights(X.shape[1], self.num_classes)
        best_weights = None
        best_val_error = float('inf')
        prev_training_error = float('inf')
        patience = 20

        t = -1
        Delta_t = 0.5
        constant_delta = 1
        m = 3
        update_delta = 0.001
        epsilon = 1e-5
        i = 0
        j = 2
        s = 0
        p = 1

        # Compute the initial acceleration and velocity
        acceleration = -(self.calculate_gradient(self.weights, X, y))
        velocity = acceleration * Delta_t / 2

        prev_weights = np.zeros(len(self.weights))
        prev_velocity = np.zeros(len(velocity))
        prev_acceleration = np.zeros(len(acceleration))

        first_loop = True
        while first_loop and t <= self.epochs:
            t += 1
            Delta_weights_norm = np.linalg.norm(velocity) * Delta_t
            if Delta_weights_norm < constant_delta:
                p = p + update_delta
                Delta_t = p * Delta_t
            
            else:
                velocity = (constant_delta * velocity) / (Delta_t * np.linalg.norm(velocity))
            
            if s >= m:
                Delta_t = Delta_t / 2
                s = 0
                self.weights = (self.weights + prev_weights) / 2
                velocity = (velocity + prev_velocity) / 4
            
            prev_weights = self.weights
            self.weights = self.weights + (velocity * Delta_t)

            second_loop = True
            while second_loop:
                prev_velocity = velocity
                prev_acceleration = acceleration

                # Create batches
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                batches = np.array_split(indices, self.num_batches)

                batch_gradients = []
                for batch_indices in batches:
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    batch_gradients.append(self.calculate_gradient(self.weights, X_batch, y_batch))
                
                acceleration = -(np.mean(batch_gradients, axis=0))
                velocity = velocity + (acceleration * Delta_t)

                if acceleration.T @ prev_acceleration > 0:
                    s = 0
                
                else:
                    s = s + 1
                    p = 1
                
                if np.linalg.norm(acceleration) > epsilon:
                    if np.linalg.norm(velocity) > np.linalg.norm(prev_velocity):
                        i = 0

                    else:
                        temp_weight = self.weights
                        self.weights = (self.weights + prev_weights) / 2
                        prev_weights = temp_weight
                        i += 1
                        
                        if i <= j:
                            velocity = (velocity + prev_velocity) / 4
                            t += 1
                            second_loop = False
                        
                        else:
                            velocity = 0
                            j = 1
                            t += 1

                if np.linalg.norm(velocity) > np.linalg.norm(prev_velocity):
                    second_loop = False

            t_error = self.calculate_error_for_weights(self.weights, X, y)
            val_error = self.calculate_error_for_weights(self.weights,X_val,y_val)
            if self.debug:
                print(f"Training error is: {t_error} and validation error is: {val_error}")
            self.training_error.append(t_error)
            self.validation_error.append(val_error)

            if val_error < best_val_error:
                best_val_error = val_error
                best_weights = self.weights

            if val_error > np.mean(self.validation_error[min(20,len(self.validation_error)):]) + np.std(self.validation_error[min(20,len(self.validation_error)):]) and prev_training_error > t_error:
                if self.debug:
                    print(f"Early stopping due to increasing validation error.")
                self.weights = best_weights
                first_loop = False
            prev_training_error = t_error

            if np.abs(np.linalg.norm(self.weights) - np.linalg.norm(prev_weights)) < 1e-6 and t > patience:
                self.weights = best_weights
                if self.debug:
                    print(f"Early stopping due to small weight changes.")
                first_loop = False

            if t == self.epochs:
                self.weights = best_weights

            if np.linalg.norm(acceleration) <= epsilon:
                first_loop = False
            
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