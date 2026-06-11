
# Goal:
# network = model.createNetwork([
#     layers.DenseLayer(input_shape, activation='sigmoid'),
#     layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
#     layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
#     layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
# ])
# model.fit(network, data_train, data_valid, loss='categoricalCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84)

import numpy as np

class activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def reluDerivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoidDerivative(x):
        s = activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmaxDerivative(x):
        s = activations.softmax(x)
        return s * (1 - s)

class weightsInitializers:
    @staticmethod
    def glorotUniform(shape):
        limit = np.sqrt(6 / sum(shape))
        return np.random.uniform(-limit, limit, size=shape)

    @staticmethod
    def heUniform(shape):
        limit = np.sqrt(6 / shape[0])
        return np.random.uniform(-limit, limit, size=shape)

class losses:
    @staticmethod
    def categoricalCrossentropy(y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    @staticmethod
    def binaryCrossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class layer:
    def __init__(self, units, activation='relu', weights_initializer='glorotUniform'):
        self.units = units
        self.activation = self.get_activation_function(activation)
        self.activationDerivative = getattr(activations, activation + 'Derivative')
        self.weights_initializer = self.get_weights_initializer(weights_initializer)
        self.weights = None
        self.biases = None
    
    def get_activation_function(self, name):
        if name not in ['relu', 'sigmoid', 'softmax']:
            raise ValueError(f"Unknown activation function: {name}")
        return getattr(activations, name)
    
    def get_weights_initializer(self, name):
        if name not in ['glorotUniform', 'heUniform']:
            raise ValueError(f"Unknown weights initializer: {name}")
        return getattr(weightsInitializers, name)
    
    def init_weights(self, input_units):
        weights_shape = (input_units, self.units)
        self.weights = self.weights_initializer(weights_shape)
        self.biases = np.zeros((1, self.units))
    
    def forward(self, input):
        weighted_sum = np.dot(input, self.weights) + self.biases
        return self.activation(weighted_sum)
    
    def backward(self, input, output, next_error_term, is_hidden_layer, learning_rate):
        if not is_hidden_layer:
            error = output - next_error_term
        else:
            error = next_error_term * self.activationDerivative(output)

        d_weights = np.dot(input.T, error)
        d_biases = np.sum(error, axis=0, keepdims=True)

        error_term_for_prev = np.dot(error, self.weights.T)

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return error_term_for_prev

class mlp:
    def __init__(self, layers):
        if len(layers) < 4:
            raise ValueError("Network must have at least 1 input layer, 2 hidden layers, and 1 output layer.")
        self.layers = self.initialize_weights_layers(layers)

    def initialize_weights_layers(self, layers):
        for i in range(len(layers) - 1):
            input_units = layers[i].units
            layers[i + 1].init_weights(input_units)
        return layers[1:]

    def fit(self, X_train, y_train, loss='categoricalCrossentropy', learning_rate=0.01, batch_size=32, epochs=10, verbose=False):
        self.X_train = X_train
        self.y_train = y_train
        self.loss_function = self.get_loss_function(loss)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        history = self.train()
        return history

    def get_loss_function(self, name):
        if name not in ['categoricalCrossentropy', 'binaryCrossentropy']:
            raise ValueError(f"Unknown loss function: {name}")
        return getattr(losses, name)

    def train(self):
        history = []
        for epoch in range(self.epochs):
            # Shuffle the training data
            indices = np.arange(self.X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = self.X_train[indices]
            y_train_shuffled = self.y_train[indices]

            epoch_loss = 0
            correct_predictions = 0

            for i in range(0, self.X_train.shape[0], self.batch_size):
                X_batch = X_train_shuffled[i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                # Forward pass
                inputs = [X_batch]
                for layer in self.layers:
                    inputs.append(layer.forward(inputs[-1]))

                # Compute loss
                loss_value = self.loss_function(y_batch, inputs[-1])
                epoch_loss += loss_value * X_batch.shape[0]
                
                # Check accuracy for this batch
                preds = np.argmax(inputs[-1], axis=1)
                labels = np.argmax(y_batch, axis=1)
                correct_predictions += np.sum(preds == labels)

                # Backward pass
                next_error_term = self.layers[-1].backward(inputs[-2], inputs[-1], y_batch, False, self.learning_rate)
                for j in range(len(self.layers) - 2, -1, -1):
                    next_error_term = self.layers[j].backward(inputs[j], inputs[j + 1], next_error_term, True, self.learning_rate)

            avg_loss = epoch_loss / self.X_train.shape[0]
            accuracy = correct_predictions / self.X_train.shape[0]
            history.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy})
            if self.verbose:
                print(f"\rEpoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.5f} - Accuracy: {accuracy:.5f}", end="")
                # print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.5f} - Accuracy: {accuracy:.5f}")
        
        return history

import pandas as pd

def main():
    data_train = pd.read_csv("data_train.csv")

    input_units = data_train.drop('Diagnosis', axis=1).shape[1]
    output_units = len(data_train['Diagnosis'].unique())

    model = mlp([
        layer(input_units),
        layer(24, activation='sigmoid', weights_initializer='glorotUniform'),
        layer(24, activation='sigmoid', weights_initializer='glorotUniform'),
        layer(output_units, activation='softmax', weights_initializer='glorotUniform')
    ])

    X_train = data_train.drop('Diagnosis', axis=1).values
    y_train = pd.get_dummies(data_train['Diagnosis']).values

    history = model.fit(
        X_train, 
        y_train, 
        loss='categoricalCrossentropy', 
        learning_rate=0.0314, 
        batch_size=10, 
        epochs=500, 
        verbose=True
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv("training_history.csv", index=False)

                
if __name__ == "__main__":
    main()