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
void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y, int epochs, std::vector<double>& lossHistory) {
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
        double currentLoss = 0.0;

        for (size_t i = 0; i < x.size(); i++) {
            double yPred = predict(x[i]);
            double error = yPred - y[i];

            dwSum += (2 / n) * error * x[i];
            dbSum += (2 / n) * error;

            currentLoss += error * error; // accumulate loss for this epoch
        }

        w -= a * dwSum;
        b -= a * dbSum;
        // register loss history
        lossHistory.push_back(currentLoss / n);
    }
}

int main() {
    std::vector<double> X, y;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> distribution(0.0, 2.0);
    std::normal_distribution<double> noise(0.0, 1.0);

    for (int i = 0; i < 100; ++i) {
        double x_val = distribution(generator);
        X.push_back(x_val);
        y.push_back(4.0 + 3.0 * x_val + noise(generator));
    }

    LinearRegression model;
    std::vector<double> loss_history;
    model.fit(X, y, 100, loss_history);

    std::ofstream results_file("results.csv");
    results_file << "X,y_true,y_pred\n";
    for(size_t i = 0; i < X.size(); ++i) {
        results_file << X[i] << "," << y[i] << "," << model.predict(X[i]) << "\n";
    }
    results_file.close();

    std::ofstream loss_file("loss_history.csv");
    loss_file << "loss\n";
    for(const auto& l : loss_history) {
        loss_file << l << "\n";
    }
    loss_file.close();
    
    std::cout << "C++ Linear Regression training finished. Results saved to CSV files." << std::endl;

    return 0;
}