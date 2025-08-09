#include "logistic_regression.h"

void init_logistic_regression(t_logistic_regression *model)
{
    model->w = 0.0;
    model->b = 0.0;
}

/**
 * Sigmoid activation function
 */
double sigmoid(double z) 
{
    return 1.0 / (1.0 + exp(-z));
}

/**
 * Hypothesis function
 */
double hypothesis(double w, double x, double b)
{
    return sigmoid(w * x + b);
}

double binary_cross_entropy(t_logistic_regression *model, double *X, double *y, int size)
{
    if (size <= 0) {
        fprintf(stderr, "Error: Invalid size\n");
        return -1;
    }

    double total_loss = 0.0;
    double eps = 1e-15;

    for (int i = 0; i < size; i++) {
        double y_pred = hypothesis(model->w, X[i], model->b);
        double clipped_pred = fmax(eps, fmin(1 - eps, y_pred));

        total_loss += -(y[i] * log(clipped_pred) + (1 - y[i]) * log(1 - clipped_pred));
    }

    return total_loss / size;
}

void fit(t_logistic_regression *model, double *X, double *y, double a, int n, int epochs, double *loss_history)
{
    if (!X || !y) {
        printf("Error: Input or output vectors are NULL\n");
        return;
    }

    init_logistic_regression(model);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double dw_sum = 0.0;
        double db_sum = 0.0;

        for (int i = 0; i < n; i++) {
            double y_pred = hypothesis(model->w, X[i], model->b);
            double error = y_pred - y[i];

            dw_sum += error * X[i];
            db_sum += error;
        }

        // Gradient descent update
        model->w -= a * dw_sum / n;
        model->b -= a * db_sum / n;

        // modularization --> finish one epoch, update loss history
        loss_history[epoch] = binary_cross_entropy(model, X, y, n);
    }
}

int main()
{
    int epochs = 1000;
    int n_samples = 100;
    t_logistic_regression model;
    model.a = 0.1; // learning rate
    double *X = (double *)malloc(n_samples * sizeof(double));
    double *y = (double *)malloc(n_samples * sizeof(double));

    double x_max = -1.0;
    for (int i = 0; i < n_samples; i++)
    {
        double x_val = rand() % 100;
        X[i] = x_val;
        y[i] = (x_val > 50.0) ? 1.0 : 0.0;
        if (x_val > x_max) {
            x_max = x_val;
        }
    }

    printf("Scaling X values (max: %f)\n", x_max);
    for (int i = 0; i < n_samples; i++) {
        X[i] = X[i] / x_max; 
    }

    double *loss_history = (double *)malloc(epochs * sizeof(double));
    if (!loss_history) {
        fprintf(stderr, "Memory allocation failed for loss history.\n");
        return 1;
    }

    fit(&model, X, y, model.a, n_samples, epochs, loss_history);

    FILE *fp = fopen("c_loss_history.csv", "w");
    if (!fp) {
        fprintf(stderr, "Error opening file for writing.\n");
        return 1;
    }

    fprintf(fp, "loss\n");

    for (int i = 0; i < epochs; i++) {
        fprintf(fp, "%f\n", loss_history[i]);
    }

    fclose(fp);

    FILE *fp_results = fopen("c_results.csv", "w");
    if (!fp_results) {
        fprintf(stderr, "Error opening file for writing.\n");
        return 1;
    }

    fprintf(fp_results, "X,y_true,y_pred\n");

    for (int i = 0; i < n_samples; i++) {
        double y_pred = hypothesis(model.w, X[i], model.b);
        fprintf(fp_results, "%f,%f,%f\n", X[i], y[i], y_pred);
    }

    fclose(fp_results);

    FILE *fp_bound = fopen("boundary.csv", "w");
    if (!fp_bound) {
        fprintf(stderr, "Error opening file for writing.\n");
        return 1;
    }

    fprintf(fp_bound, "x_boundary,y_boundary\n");
    for(int i = 0; i <= 100; ++i) {
        double x_val = i / 100.0;
        fprintf(fp_bound, "%f,%f\n", x_val, hypothesis(model.w, x_val, model.b));
    }
    fclose(fp_bound);

    // Free allocated memory
    free(X);
    free(y);
    free(loss_history);

    return 0;
}