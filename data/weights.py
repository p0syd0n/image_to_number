import math
import numpy as np

mean = 0
num_inputs = 784
num_neurons = 256

def calc_stddev(fan_in):
    return math.sqrt(2 / fan_in)

def write_biases(filename, values):
  with open(filename, "w") as file:
    for value in values:
      file.write(str(value) + " ")
    file.close()


def write_file(name, size, std_dev):
    normal_dist = np.random.normal(loc=0, scale=std_dev, size=size)
    with open(name, "w") as file:
        for value in normal_dist:
            file.write(f"{round(value, 6)} ")


biases_layer1 = ["0.5"] * num_neurons
biases_layer2 = ["0.5"] * num_neurons
biases_layersfinal = ["0.5"] * num_neurons

write_file("data/weight/weight_layer_1.txt", num_inputs*num_neurons, calc_stddev(num_inputs))
write_file("data/weight/weight_layer_2.txt", num_neurons*num_neurons, calc_stddev(num_neurons))
write_file("data/weight/weight_layer_3.txt", num_neurons*num_neurons, calc_stddev(num_neurons))

write_biases("data/bias/bias_layer_1.txt", biases_layer1)

write_biases("data/bias/bias_layer_2.txt", biases_layer2)

write_biases("data/bias/bias_layer_3.txt", biases_layersfinal)





