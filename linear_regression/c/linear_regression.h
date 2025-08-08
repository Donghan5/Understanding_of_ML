#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef struct s_linear_regression
{
    double w;
    double b;
    double a;
} t_linear_regression;


void init_model(t_linear_regression *model);
double mean_squared_error(double *y_true, double *y_pred, int n_sample);
double predict(double w, double x, double b);
void gradient_descent(t_linear_regression *model, double *X, double *y, int epochs, int n_samples, double *loss_history);


#endif