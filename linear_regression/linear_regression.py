import numpy as np
import matplotlib.pyplot as plt


"""
formula: y = Wx + b
where:
    y: target variable
    W: weights (coefficients)
    x: input features
    b: bias (intercept)
"""

class LinearRegression:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(1, 1)  # Initialize weights randomly
        self.bias = np.random.rand(1)  # Initialize bias randomly
        self.loss_history = []

    def hypothesis(self, x):   # Predict function model => y = Wx + b
        return np.dot(x, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  # Squeared error loss (**: square operator)
    
    def fit(self, x, y, epochs):
        n_samples = x.shape[0]
        y = y.reshape(-1, 1)  # Ensure y is a column vector (because Wx + b expects column vector)
        
        for epoch in range(epochs):
            y_pred = self.hypothesis(x)
            loss = self.mean_squared_error(y, y_pred)
            self.loss_history.append(loss)

            error = y_pred - y  # Calculate error

            # Calculate gradients
            dW = (2 / n_samples) * np.dot(x.T, error)
            db = (2 / n_samples) * np.sum(error)

            # Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
    
        return self.loss_history

# Visualization of results
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Generate random data, making feature data
y = 4 + 3 * X + np.random.randn(100, 1) # Add noise to the target variable  3 is Weight, 4 is Bias

model = LinearRegression(learning_rate=0.01)
losses = model.fit(X, y, epochs=100)

plt.style.use('seaborn-v0_8-whitegrid')

# Plotting the loss curve and regression line
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Main title for the entire figure
fig.suptitle('Linear Regression Model Analysis', fontsize=16)


ax1 = axes[0] # Left side canvas
ax1.plot(losses, color='dodgerblue', linewidth=2)
ax1.set_title('Model Loss over Epochs', fontsize=14)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Mean Squared Error')


ax2 = axes[1] # Right side canvas
ax2.scatter(X, y, alpha=0.6, edgecolors='k', label='Data Points') 
ax2.plot(X, model.hypothesis(X), color='crimson', linewidth=3, label='Regression Line')
ax2.set_title('Regression Fit on Data', fontsize=14)
ax2.set_xlabel('X (Feature)')
ax2.set_ylabel('y (Target)')
ax2.legend()

# Adjust layout to prevent overlap with main title
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("linear_regression_analysis.png", dpi=150) # Save with higher resolution
plt.show()
plt.close()
