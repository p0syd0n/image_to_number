import matplotlib.pyplot as plt
from time import sleep
# Read data from file
with open("accuracy.txt", "r") as f:
    contents = f.read()

# Convert space-separated values to list of floats
accuracies = list(map(float, contents.strip().split()))
# Check if we have exactly 60 values
if len(accuracies) != 60*15:
    raise ValueError(f"Expected 60 accuracy values, got {len(accuracies)}")

# X-axis: thousands of images trained (1k to 60k)
x_values = list(range(1, 60*15 +1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Model Accuracy Over Training Progress")
plt.xlabel("Thousands of Images")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig("chart.png")
print("Chart saved as chart.png")
