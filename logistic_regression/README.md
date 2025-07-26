# Logistic Regression from Scratch

This project is a from-scratch implementation of a Logistic Regression model for binary classification. It uses NumPy for all matrix operations and does not rely on any high-level machine learning libraries like PyTorch or TensorFlow.

The goal is to deeply understand the core mechanics of the algorithm, including the forward pass, loss calculation, and gradient descent.

This is a fundamental step in building more complex models from scratch, moving from a high-level framework implementation to a low-level, foundational understanding.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core Concepts
Logistic Regression is a classification algorithm that works by combining a linear equation with the sigmoid function.

'The Linear Part: z = Wx + b'

Just like in Linear Regression, the model first computes a linear combination of the input features X, weights W, and bias b.

The primary role of W and b is to learn a linear decision boundary that best separates the classes. These are the parameters that will be initialized and updated during training.

'The Sigmoid Function: σ(z) = 1 / (1 + e⁻ᶻ)'

The output of the linear equation (z) can be any real number. The sigmoid function takes this value and "squashes" it into a range between 0 and 1.

This output can be interpreted as the predicted probability of the positive class (e.g., the probability of a sample being '1').

The sigmoid function itself is a fixed mathematical formula; it does not have parameters that are learned.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Implementation Steps
The model is trained using the principle of Gradient Descent. The process involves initializing parameters and then iteratively updating them to minimize a loss function.

1.Initialization

Initialize the weights W and bias b. Typically, these are initialized with small random numbers or zeros.

2.Training Loop

The model iterates through the training data for a specified number of epochs. In each epoch, the following steps are performed:

2.1 Forward Pass: Calculate Prediction

First, compute the linear combination for all samples: z = np.dot(X, W) + b.

Then, apply the sigmoid function to get the final predicted probabilities: y_pred = sigmoid(z).

2.2 Loss Calculation

Calculate the error between the predicted probabilities (y_pred) and the true labels (y) using the Binary Cross-Entropy (BCE) Loss function. This function is ideal for binary classification tasks.

2.3 Backward Pass: Calculate Gradients

Differentiate the BCE loss function with respect to the parameters W and b. This gives us the gradients (dW and db), which indicate the direction of the steepest ascent of the loss function.

2.4 Parameter Update

Update the weights and bias by taking a small step in the opposite direction of their respective gradients. This is the core of gradient descent.

W  =  W  -  learning_rate * dW

b  =  b  -  learning_rate * db