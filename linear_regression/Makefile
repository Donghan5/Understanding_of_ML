# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# 소스 파일 및 실행 파일 이름
SRCS = LinearRegression.cpp
TARGET = linear_model
PLOT_SCRIPT = visualize_cpp.py
PLOT_OUTPUT = cpp_results_plot.png

all: $(TARGET)

# 컴파일 규칙
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
	rm -f $(TARGET) results.csv loss_history.csv $(PLOT_OUTPUT)
	rm -f linear_regression_analysis.png
	rm -f linear_regression_result.png
	@echo "Cleanup complete."

.PHONY: all run plot clean