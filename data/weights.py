import math
import numpy as np

mean = 0
std_dev = 2/784


def write_file(name, size):
  global mean, std_dev
  normal_dist = np.random.normal(loc=mean, scale=std_dev, size=size)
  with open(name, "w") as file:
    for i in range(0, len(normal_dist)):
      file.write(str(round(normal_dist[i], 6))+" ")
    file.close()

write_file("weight_layer_input.txt", 784*16)
write_file("weight_layer_1.txt", 16*16)
write_file("weight_layer_2.txt", 16*16)
