# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# 소스 파일 및 실행 파일 이름
SRCS = LogisticRegression.cpp
TARGET = logistic_model
PLOT_SCRIPT = visualize_cpp.py
PLOT_OUTPUT = cpp_results_plot.png

# 기본 규칙: 'make' 또는 'make all'을 입력하면 C++ 코드를 컴파일
all: $(TARGET)

# 컴파일 규칙
$(TARGET): $(SRCS)
	@echo "Compiling C++ source..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
	@echo "Compilation finished: $(TARGET) created."

# C++ 프로그램 실행 규칙: csv 결과 파일을 생성
run: $(TARGET)
	@echo "Running C++ model to generate CSV files..."
	./$(TARGET)

# Python 시각화 스크립트 실행 규칙
plot:
	@echo "Visualizing C++ results using $(PLOT_SCRIPT)..."
	python3 $(PLOT_SCRIPT)

# 생성된 파일 정리 규칙
clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET) results.csv loss_history.csv boundary.csv $(PLOT_OUTPUT)
	rm -f logistic_regression_analysis_corrected.png
	rm -f logistic_regression_result.png
	@echo "Cleanup complete."

.PHONY: all run plot clean