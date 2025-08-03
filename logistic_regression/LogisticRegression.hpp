#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm> // for std::max/min

class LogisticRegression {
    private:
        double w; // weight
        double b; // bias
        double a; // learning rate

    public:
        LogisticRegression();
        ~LogisticRegression();

        void fit(const std::vector<double>& X, const std::vector<int>& y, int epochs, std::vector<double>& lossHistory);
        double hypothesis(double x) const;
        double sigmoid(double z) const;
        double binary_cross_entropy(const std::vector<int>& y_true, const std::vector<double>& y_pred) const;        
        
        // Getters for w and b
        double get_weight() const { return w; }
        double get_bias() const { return b; }

};



#endif // LOGISTIC_REGRESSION_H