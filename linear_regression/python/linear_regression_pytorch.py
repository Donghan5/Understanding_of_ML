import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# creating data set
torch.manual_seed(42)
X = torch.linspace(0, 1, 100).unsqueeze(1)
y = 2 * X + 1 + 0.1 * torch.randn(X.size())

# linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input and output are 1 dim
    
    def forward(self, x):
        return self.linear(x)
    
# model declaration    
model = LinearRegression()

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training
epochs = 100
losses = []

for epochs in range(epochs):
    pred = model(X)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

# visualization

with torch.no_grad():
    y_pred = model(X)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y.numpy(), label="Real")
plt.plot(X.numpy(), y_pred.numpy(), color='red', label="Predicted")
plt.title("Linear Regression (With PyTorch)")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression_result_pytorch.png")
