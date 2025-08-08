# Linear Regression from Scratch
This project is a from-scratch implementation of a simple Linear Regression model. It uses NumPy for all numerical operations and does not rely on any high-level machine learning libraries like PyTorch or TensorFlow.

The goal is to build a deep, foundational understanding of how Linear Regression works by manually implementing the core components: the prediction formula, the loss function, and the gradient descent optimization algorithm.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Core Concepts
Linear Regression is a fundamental algorithm used to model the relationship between a dependent variable (y) and one or more independent variables (X). It aims to find the best-fitting straight line that describes the data.

The model is defined by the classic linear equation:

'Formula: y = Wx + b'

y: The predicted output value.

W (Weight): The coefficient or slope of the line. It determines how much the output y changes for a one-unit change in the input x.

x: The input feature.

b (Bias): The y-intercept of the line. It's the value of y when x is zero.

The parameters W and b are what the model learns from the data during the training process.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Implementation Steps
The model is trained using the Gradient Descent algorithm, which iteratively adjusts the model's parameters to minimize the error between predictions and actual values.

1.Initialization

Initialize the weight W and bias b. These are typically set to zero or small random values at the start.

2.Training Loop

The model iterates through the training data for a specified number of epochs. In each epoch, the following steps are performed:

2.1 Forward Pass: Calculate Prediction

Calculate the predicted output y_pred for all training samples using the linear formula: y_pred = np.dot(X, W) + b.

2.2 Loss Calculation

Measure the model's error using the Mean Squared Error (MSE) loss function. MSE calculates the average of the squared differences between the predicted values (y_pred) and the true values (y).

2.3 Backward Pass: Calculate Gradients

Calculate the gradients of the MSE loss function with respect to the parameters W and b. The gradients (dW and db) tell us how to change W and b to reduce the loss.

2.4 Parameter Update

Update the weight and bias by moving them in the opposite direction of their gradients, scaled by a learning_rate. This step minimizes the loss.

W  =  W  -  learning_rate * dW

b  =  b  -  learning_rate * db
------------------------------------------------------------------------------------------------------------------------------------------------------------------
## How to test?
1. `chmod +x auto_python_result.sh`
2. `./auto_python_result.sh`
3. To clean .csv, .png files, move each folder (c/cpp) and do `make clean`