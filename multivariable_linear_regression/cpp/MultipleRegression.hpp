#ifndef MULTIPLEREGRESSION_HPP
#define MULTIPLEREGRESSION_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

class MultipleRegression {
    private:
        std::vector<double> w; // weight (vector)
        double b; // bias
        double a; // learning rate
    
    public:
        MultipleRegression(int n_features, double learning_rate);
        ~MultipleRegression();

        double predict(const std::vector<double>& x) const;
        void SGD(const std::vector<std::vector<double>>& x, const std::vector<double>& y, int epochs);
};

#endif // MULTIPLEREGRESSION_HPP