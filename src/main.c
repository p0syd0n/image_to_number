#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <sys/types.h>
#include "../include/mnist.h"
#include "../include/utils.h"



int main(void)
{
    // layer input is initial pixels for 1 image - 28*28
    double layer_input[num_inputs] = {0};

    // An array of doubles (neurons)
    double layer_1[num_neurons_per_layer] = {0};
    double layer_2[num_neurons_per_layer] = {0};
    // Layer final is just our probability distribution, with 10 options
    double layer_final[num_outputs] = {0};
    
    // Bias coming out of layer input and layer 1 and layer 2
    double bias_layer_input[num_neurons_per_layer];
    double bias_layer_1[num_neurons_per_layer];
    double bias_layer_2[num_neurons_per_layer];
    initialize_bias(bias_layer_input, num_neurons_per_layer, 0.5);
    initialize_bias(bias_layer_1, num_neurons_per_layer, 0.5);
    initialize_bias(bias_layer_2, num_neurons_per_layer, 0.5);

    // Create weight arrays
    // for each of the next 16 neurons, you need as many weights as the previous neuron count
    double layer_input_weights[num_neurons_per_layer][num_inputs] = {0};
    double layer_1_weights[num_neurons_per_layer][num_neurons_per_layer] = {0};
    double layer_2_weights[num_neurons_per_layer][num_neurons_per_layer] = {0};

    // Load weights from data folder
    load_weights_1_from_file(layer_input_weights, "weight_layer_input.txt");
    load_weights_other_from_file(layer_1_weights, "weight_layer_1.txt");
    load_weights_other_from_file(layer_2_weights, "weight_layer_2.txt");

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
            // Layer input element is the pixel
            layer_input[j] = train_image[i][j];
        }
        // Now, layer input is loaded with the image
        printf("Loaded image %d\n", i+1);
        printf("Starting to populate layer 1\n");

        // Populate layer 1
        for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
            // Convert to GSL vectors
            gsl_vector_view v_input = gsl_vector_view_array(layer_input, num_inputs);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_input_weights[next_neuron], num_inputs);
            
            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer input
            dot_product += bias_layer_input[next_neuron];
            // Apply activation function
            dot_product = relu(dot_product);
            // Set layer 1 neuron
            layer_1[next_neuron] = dot_product;

            //printf("Layer 1 neuron %d = %lf\n", next_neuron, dot_product);
        }

        printf("Starting to populate layer 2\n");

        // Populate layer 2
        for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
            // Convert to GSL vectors
            gsl_vector_view v_input = gsl_vector_view_array(layer_1, num_neurons_per_layer);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_1_weights[next_neuron], num_neurons_per_layer);

            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer 1
            dot_product += bias_layer_1[next_neuron];
            // Apply activation function
            //printf("Dot product before relu: %lf\n", dot_product);
            dot_product = relu(dot_product);
            // Set layer 2 neuron
            layer_2[next_neuron] = dot_product;

            //printf("Layer 2 neuron %d = %lf\n", next_neuron, layer_2[next_neuron]);
        }

        printf("Starting to populate layer final\n");

        // Populate layer final
        for (int next_neuron = 0; next_neuron < num_outputs; next_neuron++) {
            // Convert to GSL vectors
            gsl_vector_view v_input = gsl_vector_view_array(layer_2, num_neurons_per_layer);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_2_weights[next_neuron], num_neurons_per_layer);
            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer 2
            dot_product += bias_layer_2[next_neuron];
            // Apply activation function
            dot_product = relu(dot_product);
            // Set layer final neuron
            layer_final[next_neuron] = dot_product;

            //printf("Layer final neuron %d = %lf\n", next_neuron, dot_product);
        }

        // Time to softmax
        double total = 0;
        for (int neuron = 0; neuron < num_outputs; neuron++) {
            total += exp(layer_final[neuron]);
        }

        printf("total of neuron final prior to softmax: %f\n", total);

        for (int neuron_ = 0; neuron_ < num_outputs; neuron_++) {
            layer_final[neuron_] = (exp(layer_final[neuron_]) / total);
            //printf("Softmaxed: %f\n", layer_final[neuron_]);
        }

        total = 0;
        for (int neuron = 0; neuron < num_outputs; neuron++) {
            total += layer_final[neuron];
        }
        printf("total post softmax: %f\n", total);
        break;
        
    }
    return 0;
}
/* 
    TODO:
        - rename bias/weight/file variables to make more sense
        - modularize each layer step? 
*/