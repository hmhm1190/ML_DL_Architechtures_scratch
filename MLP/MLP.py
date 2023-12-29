# mlp.py

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Multi-Layer Perceptron (MLP) with random weights and biases.

        Parameters:
        - input_size: Number of features in the input data.
        - hidden_size: Number of neurons in the hidden layer.
        - output_size: Number of classes in the output layer.
        """
        # Initialize weights with random values from a normal distribution
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        """
        Perform the forward pass of the MLP.

        Parameters:
        - x: Input data.

        Returns:
        - Output probabilities after the forward pass.
        """
        hidden, hidden_activation, output, output_probs = self.forward_pass(x)
        return output_probs

    def backward(self, x, y_true, learning_rate=0.01):
        """
        Perform the backward pass (backpropagation) to update weights and biases.

        Parameters:
        - x: Input data.
        - y_true: True labels.
        - learning_rate: Learning rate for weight updates.

        Returns:
        - Loss after the backward pass.
        """
        hidden, hidden_activation, output, output_probs = self.forward_pass(x)
        loss = self.categorical_cross_entropy(output_probs, y_true)
        output_grad = self.calculate_output_gradient(output_probs, y_true)
        self.update_weights_backward(x, hidden, hidden_activation, output_grad, learning_rate)
        return loss

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01):
        """
        Train the MLP on the given training data.

        Parameters:
        - x_train: Input training data.
        - y_train: True labels for training data.
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for weight updates.
        """
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_train, y_train):
                x = x.reshape(1, -1)
                y_true = y_true.reshape(1, -1)
                total_loss += self.backward(x, y_true, learning_rate)
            print(f'Epoch {epoch}, Loss: {total_loss / len(x_train)}')

    def predict(self, x):
        """
        Make predictions on new data.

        Parameters:
        - x: Input data for prediction.

        Returns:
        - Predicted labels.
        """
        return np.argmax(self.forward(x), axis=1)

    def forward_pass(self, x):
        """
        Perform the forward pass of the MLP.

        Parameters:
        - x: Input data.

        Returns:
        - Tuple containing hidden layer, hidden layer activation, output layer, and output probabilities.
        """
        hidden = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        hidden_activation = self.relu(hidden)
        output = np.dot(hidden_activation, self.weights_hidden_output) + self.bias_output
        output_probs = self.softmax(output)
        return hidden, hidden_activation, output, output_probs

    def update_weights_backward(self, x, hidden, hidden_activation, output_grad, learning_rate):
        """
        Update weights and biases in the backward pass.

        Parameters:
        - x: Input data.
        - hidden: Hidden layer values from the forward pass.
        - hidden_activation: Activation values of the hidden layer.
        - output_grad: Gradient of the loss with respect to the output.
        - learning_rate: Learning rate for weight updates.
        """
        hidden_grad = np.dot(output_grad, self.weights_hidden_output.T)
        hidden_grad[hidden <= 0] = 0  # ReLU derivative

        self.weights_hidden_output -= learning_rate * np.dot(hidden_activation.T, output_grad)
        self.bias_output -= learning_rate * np.sum(output_grad, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(x.T, hidden_grad)
        self.bias_hidden -= learning_rate * np.sum(hidden_grad, axis=0, keepdims=True)

    def relu(self, x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
        - x: Input to the ReLU function.

        Returns:
        - Output after applying the ReLU activation.
        """
        return np.maximum(0, x)

    def softmax(self, x):
        """
        Softmax activation function.

        Parameters:
        - x: Input to the softmax function.

        Returns:
        - Output after applying the softmax activation.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def categorical_cross_entropy(self, output_probs, y_true):
        """
        Calculate categorical cross-entropy loss.

        Parameters:
        - output_probs: Predicted probabilities from the model.
        - y_true: True labels.

        Returns:
        - Categorical cross-entropy loss.
        """
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(output_probs + 1e-10)) / m

    def calculate_output_gradient(self, output_probs, y_true):
        """
        Calculate the gradient of the loss with respect to the output.

        Parameters:
        - output_probs: Predicted probabilities from the model.
        - y_true: True labels.

        Returns:
        - Gradient of the loss with respect to the output.
        """
        m = y_true.shape[0]
        return (output_probs - y_true) / m
