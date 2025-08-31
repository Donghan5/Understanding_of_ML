import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Creating Data
n_samples = 1000
n_features = 1

X = np.random.randn(n_samples, n_features)

true_weights = np.array([2.0, 3.0, -1.0])
true_bias = 5.0

noise = np.random.randn(n_samples) * 0.1

# Formula => y = w1 * x1 + w2 * x2 + ... + wn * xn + b
y = X @ true_weights + true_bias + noise

# Train / test split
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

class MultipleLinearRegression:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)

    def forward(self, x):
        return x @ self.weights + self.bias

    def loss_function(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, x, y, y_pred):
        n = x.shape[0]
        dw = (-2/n) * (x.T @ (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        return dw, db

    def SGD(self, x, y, learning_rate):
        y_pred = self.forward(x)
        loss = self.loss_function(y, y_pred)
        
        dw, db = self.backward(x, y, y_pred)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return loss

model = MultipleLinearRegression(n_features)
criterion = model.loss_function(y, model.forward(X))
optimizer = model.SGD(X, y, 0.01)

n_epochs = 100
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    train_loss = optimizer
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}")
