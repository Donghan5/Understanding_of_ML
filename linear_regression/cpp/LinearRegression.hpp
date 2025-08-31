#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

#include <vector>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <random>

class LinearRegression {
    
    private:
        double w;   // weight (slope)
        double b;   // bias (intercept)
        double a;   // learning rate


    public:
        LinearRegression();
        ~LinearRegression();
        void gradientDescent(const std::vector<double>& x, const std::vector<double>& y, int epochs, std::vector<double>& lossHistory);
        double predict(double x) const;
        double meanSquaredError(const std::vector<double>& yTrue, const std::vector<double>& yPred) const;
};

#endif // LINEAR_REGRESSION_HPP