#!/bin/bash
echo "================================================================================"
echo "Welcome to the automated Python script execution for Logistic Regression."
echo "================================================================================"

echo "Moving upper directory to execute Python virtual environment..."
cd ..

echo "Activating Python Virtual environment..."
source .venv/bin/activate

echo "Installing Pandas and Matplotlib..."
pip install pandas matplotlib

echo "Moving to logistic_regression directory..."
cd logistic_regression

echo "Compiling C++ code..."
make

echo "Running C++ code... and make visualization"
./logistic_model
make plot

echo "Running Python visualization script..."
python3 logistic_regression.py
python3 logistic_regression_pytorch.py

echo "All Python scripts executed successfully."