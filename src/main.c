#include <stdio.h>
#include <time.h>
#include <math.h>

#include "../include/mnist.h"

#define inputs_per_neuron 784 // 28 * 28
#define num_neurons_per_layer 16

int main(void)
{
    double layer_1_weights[num_neurons_per_layer][inputs_per_neuron];
    double layer_2_weights[num_neurons_per_layer][inputs_per_neuron];

    srand(time(NULL));
    
    
    load_mnist();

    printf("%d\n", train_label[0]);

    return 0;
}