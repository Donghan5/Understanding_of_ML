C = gcc
CFLAGS = -Wall -Wextra -Werror
LDLIBS = -lm

SRCS = logistic_regression.c
TARGET = logistic_model
PLOT_SCRIPT = visualize_c.py
PLOT_OUTPUT = c_results_plot.png

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling C source..."
	$(C) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDLIBS)
	@echo "Compilation finished: $(TARGET) created."

run: $(TARGET)
	@echo "Running C model to generate CSV files..."
	./$(TARGET)

plot:
	@echo "Visualizing C results using $(PLOT_SCRIPT)..."
	python3 $(PLOT_SCRIPT)

clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET) boundary.csv c_results.csv c_loss_history.csv $(PLOT_OUTPUT)
	rm -f logistic_regression_analysis.png
	rm -f logistic_regression_result.png
	@echo "Cleanup complete."

.PHONY: all run plot clean