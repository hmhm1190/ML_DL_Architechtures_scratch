from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from MLP import MLP
import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# Load the Digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the labels (not needed for this example, but the MLP expects one-hot encoding)
num_classes = len(set(y))
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

# Define the MLP model
input_size = X_train.shape[1]
hidden_size = 64
output_size = num_classes
mlp = MLP(input_size, hidden_size, output_size)

# Lists to store training and validation accuracy
train_accuracy_history = []
val_accuracy_history = []

# Train the MLP and monitor overfitting/underfitting
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for x, y_true in zip(X_train, y_train_one_hot):
        x = x.reshape(1, -1)
        y_true = y_true.reshape(1, -1)
        total_loss += mlp.backward(x, y_true, learning_rate=0.012)

    # Calculate training accuracy
    y_train_pred = mlp.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy_history.append(train_accuracy)

    # Calculate validation accuracy
    y_val_pred = mlp.predict(X_test)
    val_accuracy = accuracy_score(y_test, y_val_pred)
    val_accuracy_history.append(val_accuracy)
    print(f'Epoch {epoch}, Loss: {total_loss / len(X_train)}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}')


# check_overfitting_underfitting by plotting training and validation accuracy over epochs

# Get the current date and time
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Plotting training and validation accuracy over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), train_accuracy_history, label='Training Accuracy')
plt.plot(range(epochs), val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

# Save the plot with a filename containing the date and time
filename = f"accuracy_plot_{current_datetime}.png"
plt.savefig(os.path.join("MLP/plots",filename))

# Display the plot
plt.show()

print(f"Accuracy plot saved as {filename}")
