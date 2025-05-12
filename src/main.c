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
    
    
    FILE *fptr = fopen("../data/weight_layer_1.txt", "r");
    if (fptr == NULL) {
    perror("File Open");
    return -1;
    }
    srand(time(NULL));  
    char temp_buffer[50000];
    double temp_number;
    while(fscanf(fptr, "%f", temp_buffer, &temp_number)>0) {
      printf("Read: %f\n", temp_number, temp_number);
    }
    load_mnist();

    printf("%d\n", train_label[0]);

    return 0;
}
