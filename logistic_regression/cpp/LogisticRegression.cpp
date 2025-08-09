#include "LogisticRegression.hpp"

LogisticRegression::LogisticRegression() : a(0.1) {}

LogisticRegression::~LogisticRegression() {}

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

double LogisticRegression::hypothesis(double x) const {
    return sigmoid(w * x + b);
}

// Just computing binary cross entropy, I will use it after.
// currently not using in this codes. Keep it now
double LogisticRegression::binaryCrossEntropy(const std::vector<int>& y_true, const std::vector<double>& y_pred) const {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Vectors must be of the same size for binary cross-entropy calculation.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double eps = 1e-15; // to avoid log(0)
        double clipped_pred = std::max(eps, std::min(1 - eps, y_pred[i]));
        loss += -(y_true[i] * std::log(clipped_pred) + (1 - y_true[i]) * std::log(1 - clipped_pred));
    }

    double meanLoss = loss / y_true.size();
    return meanLoss;
}

void LogisticRegression::fit(const std::vector<double>& X, const std::vector<int>& y, int epochs, std::vector<double>& lossHistory) {
    if( X.size() != y.size()) {
        throw std::invalid_argument("Input and output vectors must be of the same size.");
    }
    
    w = 0.0;
    b = 0.0;

    const double n = static_cast<double>(X.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double dWSum = 0.0;
        double dbSum = 0.0;
        double currentLoss = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double yPred = hypothesis(X[i]);
            double error = yPred - y[i];

            dWSum += error * X[i];
            dbSum += error;

            // Calculate current loss with binary cross entropy
            double eps = 1e-15; // to avoid log(0)
            double clipped_pred = std::max(eps, std::min(1 - eps, yPred));
            currentLoss += -(y[i] * std::log(clipped_pred) + (1 - y[i]) * std::log(1.0 - clipped_pred));
        }
        
        // Update weights and bias
        dWSum /= n;
        dbSum /= n;
        
        
        w -= a * dWSum;
        b -= a * dbSum;
        
        lossHistory.push_back(currentLoss / n);
    }
}


int main() {
    std::vector<double> X;
    std::vector<int> y;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> distribution(0.0, 2.0);
    
    for (int i = 0; i < 100; ++i) {
        double x_val = distribution(generator);
        X.push_back(x_val);
        y.push_back(x_val > 1.0 ? 1 : 0);
    }

    LogisticRegression model;
    std::vector<double> lossHistory;
    model.fit(X, y, 1000, lossHistory);

    std::ofstream results_file("results.csv");
    results_file << "X,y_true\n";
    for(size_t i = 0; i < X.size(); ++i) {
        results_file << X[i] << "," << y[i] << "\n";
    }
    results_file.close();

    std::ofstream boundary_file("boundary.csv");
    boundary_file << "x_boundary,y_boundary\n";
    for(int i=0; i<=100; ++i) {
        double x_val = 2.0 * i / 100.0;
        boundary_file << x_val << "," << model.hypothesis(x_val) << "\n";
    }
    boundary_file.close();

    std::ofstream loss_file("loss_history.csv");
    loss_file << "loss\n";
    for(const auto& l : lossHistory) {
        loss_file << l << "\n";
    }
    loss_file.close();
    
    std::cout << "C++ Logistic Regression training finished. Results saved to CSV files." << std::endl;
    return 0;
}