#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct s_logistic_regression {
    double a; // learning rate
    double w; // weight
    double b; // bias
    int n;
} t_logistic_regression;

double sigmoid(double z);
double binary_cross_entropy(t_logistic_regression *model, double *X, double *y, int size);
void fit(t_logistic_regression *model, double *X, double *y, double a, int n, int epochs, double *loss_history);
double hypothesis(double w, double x, double b);


#endif // LOGISTIC_REGRESSION_H