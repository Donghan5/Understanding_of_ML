import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Creating Data
n_samples = 1000
n_features = 1

# Generate input data
X = torch.randn(n_samples, n_features)

# Formula y = w1 * x1 + w2 * x2 + ... + wn * xn + b
true_weights = torch.tensor([2.0, 3.0, -1.0])
true_bias = 5.0

noise = torch.randn(n_samples) * 0.1

y = X @ true_weights + true_bias + noise

train_size = int(0.8 * n_samples)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]


# Using nn.Linear (Method 1)
class MultipleLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

# Define the parameter (Method 2)
class ManualLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(ManualLinearRegression, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x @ self.weights + self.bias

model = MultipleLinearRegression(n_features)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # Using Stochastic Gradient Descent

n_epochs = 100
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    y_pred = model(X_train)
    train_loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_losses.append(train_loss.item())

    # Validation
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = criterion(y_test_pred, y_test)
        
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

print("\n=== Trained parameters ===")
print(f"Trained weights: {model.linear.weight.data.squeeze().numpy()}")
print(f"Trained bias: {model.linear.bias.data.item():.4f}")
print(f"\nTrue weights: {true_weights.numpy()}")
print(f"True bias: {true_bias:.4f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.legend()

plt.subplot(1, 2, 2)
with torch.no_grad():
    y_pred_all = model(X_test)
    plt.scatter(y_test.numpy(), y_pred_all.numpy(), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()
