import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)
n_samples = 100
X = torch.rand(n_samples, 1) * 10
y = (X > 5).float() # if element is greater than 5, it is 1, else 0 

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))     # basic of logistic regression is using sigmoid
    
model = LogisticRegression()

criterion = nn.BCELoss()  # binary cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = []

for epoch in range(200):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())

with torch.no_grad():
    x_test = torch.linspace(0, 10, 100).unsqueeze(1)
    y_test = model(x_test)

plt.figure(figsize=(12, 5))

# loss curve
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# prediction result
plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y.numpy(), label="Real")
plt.plot(x_test.numpy(), y_test.numpy(), color='red', label="Predicted")
plt.title("Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("logistic_regression_result_pytorch.png")
