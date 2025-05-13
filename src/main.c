#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <sys/types.h>
#include "../include/mnist.h"
#include "../include/utils.h"



int main(void)
{
    // layer 1 is initial pixels for 1 image - 28*28
    double layer_1[num_inputs] = {0};

    // Layer 2 is the neuron count per layer - 16
    double layer_2[num_neurons_per_layer] = {0};
    double layer_3[num_neurons_per_layer] = {0};
    
    // There is bias for layers 1 and 2 so far (going out of layer 1 and 2)
    double bias_layer_1[num_neurons_per_layer];
    double bias_layer_2[num_neurons_per_layer];

    for (int i = 0; i < num_neurons_per_layer; i++) {
        bias_layer_1[i] = 0.5;
    }
    
    for (int i = 0; i < num_neurons_per_layer; i++) {
        bias_layer_2[i] = 0.5;
    }
    

    // Create weight arrays
    // for each of the next 16 neurons, you need as many weights as the previous neuron count
    double layer_1_weights[num_neurons_per_layer][num_inputs] = {0};
    double layer_2_weights[num_neurons_per_layer][num_neurons_per_layer] = {0};
    double layer_3_weights[num_neurons_per_layer][num_neurons_per_layer] = {0};

    // Load weights from data folder
    load_weights_1_from_file(layer_1_weights, "weight_layer_1.txt");
    load_weights_other_from_file(layer_2_weights, "weight_layer_2.txt");
    load_weights_other_from_file(layer_3_weights, "weight_layer_3.txt");

    // Load mnist database
    load_mnist();

    // amount of training images (total bytes / size of one  image)
    size_t train_images_count = sizeof(train_image) / sizeof(train_image[0]);
    
    // For each image
    for (int i = 0; i < train_images_count; i+= 1) {
        // Size of an image (total bytes of image / size of one pixel(double))
        size_t length_of_image = sizeof(train_image[i]) / sizeof(double);

        // For each pixel:
        for (int j = 0; j < length_of_image; j++) {
            // Layer 1 element is the pixel
            layer_1[j] = train_image[i][j];
        }
        // Now, layer 1 is loaded with the image
        printf("Loaded image %d\n", i+1);
        printf("Starting to populate layer 2\n");

        // Populate layer 2
        for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
            // Convert to GSL vectors
            gsl_vector_view v_input = gsl_vector_view_array(layer_1, num_inputs);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_1_weights[next_neuron], num_inputs);
            
            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer 1
            dot_product += bias_layer_1[next_neuron];
            // Apply activation function
            dot_product = relu(dot_product);
            // Set layer 2 neuron
            layer_2[next_neuron] = dot_product;

            printf("Layer 2 neuron %d = %lf\n", next_neuron, dot_product);
        }
        break;
    }


    return 0;
}
/* 
    TODO:
        - rename bias/weight/file variables to make more sense
        - modularize each layer step? 
*/