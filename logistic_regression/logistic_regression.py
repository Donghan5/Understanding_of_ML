import numpy as np
import matplotlib.pyplot as plt

"""Implementation of the logistic regression model."""

class LogisticRegression:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, 1)  # Initialize weights randomly
        self.bias = np.random.rand(1)  # Initialize bias randomly
        self.loss_history = []


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hypothesis(self, x):
        x = np.dot(x, self.weights) + self.bias
        return self.sigmoid(x)

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15 # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # limit range of predictions
        
        # Binary cross entropy loss
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def fit(self, x, y, epochs):
        n_samples = x.shape[0]
        y = y.reshape(-1, 1)

        for epoch in range(epochs):
            y_pred = self.hypothesis(x)
            loss = self.binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y

            # derivatives
            dW = (1 / n_samples) * np.dot(x.T, error)
            db = (1 / n_samples) * np.sum(error)

            # update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

        return self.loss_history

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
# X가 1을 기준으로 0과 1로 나뉘도록 y 생성
y = (X > 1).astype(int)

model = LogisticRegression(learning_rate=0.1)
losses = model.fit(X, y, epochs=1000)

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Logistic Regression Model Analysis', fontsize=16)

ax1 = axes[0]
ax1.plot(losses, color='dodgerblue', linewidth=2)
ax1.set_title('BCE Loss over Epochs', fontsize=14)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Binary Cross-Entropy Loss')

ax2 = axes[1]
ax2.scatter(X[y==0], y[y==0], alpha=0.7, edgecolors='k', label='Class 0')
ax2.scatter(X[y==1], y[y==1], alpha=0.7, edgecolors='k', label='Class 1')

x_boundary = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_boundary = model.hypothesis(x_boundary)
ax2.plot(x_boundary, y_boundary, color='crimson', linewidth=3, label='Predicted Probability')
ax2.axhline(y=0.5, color='gray', linestyle='--', label='Threshold (0.5)') # 0.5 기준선

ax2.set_title('Logistic Regression Fit', fontsize=14)
ax2.set_xlabel('X (Feature)')
ax2.set_ylabel('Probability (y_pred)')
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("logistic_regression_analysis_corrected.png", dpi=150)
plt.show()
plt.close()
