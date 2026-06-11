import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def fit(self, X_train, y_train, X_valid, y_valid, loss='categoricalCrossentropy', learning_rate=0.01, batch_size=32, epochs=10, verbose=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
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
            
            train_avg_loss, train_avg_accuracy = self.trainEpoch(X_train_shuffled, y_train_shuffled)
            valid_loss, valid_accuracy = self.validationRun()

            history.append({
                'epoch': epoch, 
                'train_loss': train_avg_loss, 
                'train_accuracy': train_avg_accuracy, 
                'val_loss': valid_loss, 
                'val_accuracy': valid_accuracy
            })
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {train_avg_loss:.5f} - Valid_loss: {valid_loss:.5f} - Accuracy: {train_avg_accuracy:.5f} - Valid_accuracy: {valid_accuracy:.5f}")
            else:
                up = "\033[2A" if epoch > 0 else ""
                print(f"{up}\rEpoch {epoch + 1}/{self.epochs}\033[K\n"
                    + f" - Loss: {train_avg_loss:.5f} - Valid_loss: {valid_loss:.5f}\033[K\n"
                    + f" - Accuracy: {train_avg_accuracy:.5f} - Valid_accuracy: {valid_accuracy:.5f}\033[K", end="")
                if epoch == self.epochs - 1:
                    print()
        
        return history

    def trainEpoch(self, X_train, y_train):
        epoch_loss = 0
        correct_predictions = 0

        for i in range(0, self.X_train.shape[0], self.batch_size):
            X_batch = X_train[i:i + self.batch_size]
            y_batch = y_train[i:i + self.batch_size]
            
            batch_loss, train_accuracy = self.trainBatch(X_batch, y_batch)
            epoch_loss += batch_loss
            correct_predictions += train_accuracy

        train_avg_loss = epoch_loss / self.X_train.shape[0]
        train_avg_accuracy = correct_predictions / self.X_train.shape[0]        
        return train_avg_loss, train_avg_accuracy

    def trainBatch(self, X_batch, y_batch):
        # Forward pass
        inputs = [X_batch]
        for layer in self.layers:
            inputs.append(layer.forward(inputs[-1]))

        # Backward pass
        next_error_term = self.layers[-1].backward(inputs[-2], inputs[-1], y_batch, False, self.learning_rate)
        for j in range(len(self.layers) - 2, -1, -1):
            next_error_term = self.layers[j].backward(inputs[j], inputs[j + 1], next_error_term, True, self.learning_rate)

        # Calculate batch loss and accuracy
        batch_loss = self.loss_function(y_batch, inputs[-1]) * X_batch.shape[0]
        correct_predictions = self.compute_accuracy(inputs[-1], y_batch)
        return batch_loss, correct_predictions

    def validationRun(self):
        valid_inputs = [self.X_valid]
        for layer in self.layers:
            valid_inputs.append(layer.forward(valid_inputs[-1]))

        valid_loss = self.loss_function(self.y_valid, valid_inputs[-1])
        valid_accuracy = self.compute_accuracy(valid_inputs[-1], self.y_valid, False)
        return valid_loss, valid_accuracy

    def compute_accuracy(self, y_pred, y_true, for_batch=True):
        preds = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        if for_batch:
            return np.sum(preds == labels)
        return np.mean(preds == labels)
    
    def show_history(self, history):
        history_df = pd.DataFrame(history)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy')
        plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    data_train = pd.read_csv("data_train.csv")
    data_valid = pd.read_csv("data_valid.csv")

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
    X_valid = data_valid.drop('Diagnosis', axis=1).values
    y_valid = pd.get_dummies(data_valid['Diagnosis']).values

    history = model.fit(
        X_train, 
        y_train, 
        X_valid, 
        y_valid,
        loss='categoricalCrossentropy', 
        learning_rate=0.01, 
        batch_size=8, 
        epochs=100
        # verbose=True
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv("train_history.csv", index=False)

    model.show_history(history)

                
if __name__ == "__main__":
    main()