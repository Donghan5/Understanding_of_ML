# C
C = gcc
CFLAGS = -Wall -Wextra -Werror
LDLIBS = -lm

# 소스 파일 및 실행 파일 이름
SRCS = linear_regression.c
TARGET = linear_model
PLOT_SCRIPT = visualize_c.py
PLOT_OUTPUT = c_results_plot.png

all: $(TARGET)

# 컴파일 규칙
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
	rm -f $(TARGET) c_results.csv c_loss_history.csv $(PLOT_OUTPUT)
	rm -f linear_regression_analysis.png
	rm -f linear_regression_result.png
	@echo "Cleanup complete."

.PHONY: all run plot clean