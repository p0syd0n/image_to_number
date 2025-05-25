import math
import numpy as np

mean = 0
std_dev = 2/784
num_neurons = 16
def write_biases(filename, values):
  with open(filename, "w") as file:
    for value in values:
      file.write(str(value) + " ")
    file.close()


def write_file(name, size):
  global mean, std_dev
  normal_dist = np.random.normal(loc=mean, scale=std_dev, size=size)
  with open(name, "w") as file:
    for i in range(0, len(normal_dist)):
      file.write(str(round(normal_dist[i], 6))+" ")
    file.close()

write_file("data/weight/weight_layer_1.txt", 784*num_neurons)
write_file("data/weight/weight_layer_3.txt", num_neurons*num_neurons)


biases_layer1 = ["0.5"] * num_neurons
biases_layersfinal = ["0.5"] * num_neurons

write_biases("data/bias/bias_layer_1.txt", biases_layer1)
write_biases("data/bias/bias_layer_3.txt", biases_layersfinal)

