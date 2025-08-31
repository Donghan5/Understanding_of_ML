CXX = g++
CXXFLAGS = -std=c++11 -Wall

SRCS = LogisticRegression.cpp
TARGET = logistic_model
PLOT_SCRIPT = visualize_cpp.py
PLOT_OUTPUT = cpp_results_plot.png

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling C++ source..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
	@echo "Compilation finished: $(TARGET) created."

run: $(TARGET)
	@echo "Running C++ model to generate CSV files..."
	./$(TARGET)

plot:
	@echo "Visualizing C++ results using $(PLOT_SCRIPT)..."
	python3 $(PLOT_SCRIPT)

clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET) results.csv loss_history.csv boundary.csv $(PLOT_OUTPUT)
	rm -f logistic_regression_analysis_corrected.png
	rm -f logistic_regression_result.png
	@echo "Cleanup complete."

.PHONY: all run plot clean