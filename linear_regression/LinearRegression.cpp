#include "LinearRegression.hpp"

/**
 * Implementation of a simple linear regression model in c++.
 * formula: y = Wx + b
 */

LinearRegression::LinearRegression() : a(0.01) {}

LinearRegression::~LinearRegression() {}

/**
 *  Calculate mean squared error 
 *  @param yTrue: true values
 *  @param yPred: predicted values
 *  @return: mean squared error
 */
double LinearRegression::meanSquaredError(const std::vector<double>& yTrue, const std::vector<double>& yPred) const {
    double mse = 0.0;
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors must be of the same size for mean squared error calculation.");
    }

    for (size_t i = 0; i < yTrue.size(); i++)
    {
        double error = yTrue[i] - yPred[i];
        mse += error * error;
    }

    mse /= yTrue.size();
    return mse;
}

/**
 * Predict the output for a given input using the learned parameters.
 * @param x: input feature
 * @return: predicted output
 */
double LinearRegression::predict(double x) const {
    return w * x + b;
}

/**
 * Fit the linear regression model to the training data.
 * @param x: input features
 * @param y: target values
 * @param epochs: number of iterations for training
 */
void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y, int epochs) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Input and output vectors must be of the same size.");
    }

    // Initialize weights and bias
    w = 0.0;
    b = 0.0;
    const double n = static_cast<double>(x.size());

    // Gradient descent
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double dwSum = 0.0;
        double dbSum = 0.0;

        for (size_t i = 0; i < x.size(); i++) {
            double yPred = predict(x[i]);
            double error = yPred - y[i];

            dwSum += (2 / n) * error * x[i];
            dbSum += (2 / n) * error;
        }

        w -= a * dwSum;
        b -= a * dbSum;
    }
}


int main() {
    LinearRegression model;
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 3, 4, 5, 6};
    model.fit(x, y, 1000);
    double prediction = model.predict(6);
    std::cout << "Prediction for input 6: " << prediction << std::endl;
    return 0;
}