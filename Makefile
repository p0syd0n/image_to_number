# Variables
BUILD_DIR = build
EXECUTABLE = NeuralNetwork
SOURCES = src/main.c src/utils.c
HEADERS = include/utils.h include/mnist.h

# Default target
all: $(BUILD_DIR)/$(EXECUTABLE)

# Rebuild if sources or headers change
$(BUILD_DIR)/$(EXECUTABLE): $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

run: all
	./$(BUILD_DIR)/$(EXECUTABLE)

train: all
	echo "Generating starter weights"
	python3 /data/weights.py
	./$(BUILD_DIR)/$(EXECUTABLE) 1

test: all
	./$(BUILD_DIR)/$(EXECUTABLE) 2

image: all
	./$(BUILD_DIR)/$(EXECUTABLE) 3 file.txt

clean:
	rm -rf $(BUILD_DIR)

# test: all
# 	@echo "Running tests..."

.PHONY: all run clean test
