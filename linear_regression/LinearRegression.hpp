#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>

class LinearRegression {
    
    private:
        double w;   // weight (slope)
        double b;   // bias (intercept)
        double a;   // learning rate


    public:
        LinearRegression();
        ~LinearRegression();
        void fit(const std::vector<double>& x, const std::vector<double>& y, int epochs = 1000);
        double predict(double x) const;
        double meanSquaredError(const std::vector<double>& yTrue, const std::vector<double>& yPred) const;
};

#endif // LINEAR_REGRESSION_H