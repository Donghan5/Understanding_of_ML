#include "linear_regression.h"


void init_model(t_linear_regression *model)
{
    if (!model) return;

    model->w = 0.0;
    model->b = 0.0;
}

/**
 * Calculate the mean squared error between y_true and y_pred
 * @param y_true: true values
 * @param y_pred: predicted values
 * @return: mean squared error
 */
double mean_squared_error(double *y_true, double *y_pred, int n_sample) 
{
    double mse = 0.0;

    if (n_sample == 0)
        return 0.0;

    for (int i = 0; i < n_sample; i++) {
        double error = y_true[i] - y_pred[i];
        mse += pow(error, 2);
    }

    return mse / n_sample;
}

/**
 * @brief Predict the output for a given input using the linear regression model
 * @param w: weight
 * @param x: input feature
 * @param b: bias
 * @return: predicted output
 */
double predict(double w, double x, double b)
{
    return w * x + b;
}

/**
 * @brief Fitting model with gradient descent
 * @param model: linear regression model
 * @param learning_rate: learning rate for gradient descent
 * @param epochs: number of iterations for gradient descent
 */
void gradient_descent(t_linear_regression *model, double *X, double *y, int epochs, int n_samples, double *loss_history)
{
    if (!model || !loss_history) {
        fprintf(stderr, "Model or loss history vector is not initialized.\n");
        return;
    }

    init_model(model);
   
    for (int epoch = 0; epoch < epochs; epoch++) {
        double dw_sum = 0.0;
        double db_sum = 0.0;
        double current_loss = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double y_pred = predict(model->w, X[i], model->b);
            double error = y_pred - y[i];

            dw_sum += error * X[i];
            db_sum += error;

            current_loss += error * error; // accumulate loss for this epoch
        }

        model->w -= model->a * (dw_sum / n_samples);
        model->b -= model->a * (db_sum / n_samples);
        loss_history[epoch] = current_loss / n_samples;  // store the loss for this epoch
    }
}

int main()
{
    int epochs = 1000;
    int n_samples = 100;
    t_linear_regression model;
    model.a = 0.00001; // learning rate
    double *X = (double *)malloc(n_samples * sizeof(double));
    double *y = (double *)malloc(n_samples * sizeof(double));

    for (int i = 0; i < n_samples; i++)
    {
        double x_val = rand() % 100;
        double y_val = 2.0 * x_val + 1.0 + (rand() % 10 - 5);
        X[i] = x_val;
        y[i] = y_val;
    }

    double *loss_history = (double *)malloc(epochs * sizeof(double));
    if (!loss_history) {
        fprintf(stderr, "Memory allocation failed for loss history.\n");
        return 1;
    }

    gradient_descent(&model, X, y, epochs, n_samples, loss_history);

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
        double y_pred = predict(model.w, X[i], model.b);
        fprintf(fp_results, "%f,%f,%f\n", X[i], y[i], y_pred);
    }

    fclose(fp_results);

    // Free allocated memory
    free(X);
    free(y);
    free(loss_history);

    return 0;
}