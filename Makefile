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

clean:
	rm -rf $(BUILD_DIR)

test: all
	@echo "Running tests..."

.PHONY: all run clean test
