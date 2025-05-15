import math
import numpy as np

mean = 0
std_dev = 2/784

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

write_file("weight/weight_layer_1.txt", 784*16)
write_file("weight/weight_layer_2.txt", 16*16)
write_file("weight/weight_layer_final.txt", 16*16)


biases_layer1 = ["0.5"] * 784
biases_layers2final = ["0.5"] * 16

write_biases("bias/bias_layer_1.txt", biases_layer1)
write_biases("bias/bias_layer_2.txt", biases_layers2final)
write_biases("bias/bias_layer_final.txt", biases_layers2final)

