#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <sys/types.h>
#include <stdarg.h>
#include "../include/mnist.h"
#include "../include/utils.h"

#define LOG_BUFFER_SIZE 1024000
#define learning_rate 0.01
char log_buffer[LOG_BUFFER_SIZE];
size_t log_index = 0;

// Buffer logging system
void log_fast(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_index += vsnprintf(&log_buffer[log_index], LOG_BUFFER_SIZE - log_index, format, args);
    va_end(args);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("./file <mode> [filename]");
        exit(-1);
    }

    // Strings for future filenames - they get decided based off of what mode we launch in
    char *bias_layer_1_filename;
    char *bias_layer_3_filename;
    char *weight_layer_1_filename;
    char *weight_layer_3_filename;
    char* input_filename;

    int mode = atoi(argv[1]);
    printf("Mode: %d Input: %s\n", mode, input_filename);
    // Mode 1: training mode
    if (mode == 1) {
        bias_layer_1_filename = (char *)malloc(strlen("bias_layer_1.txt")+1);
        strcpy(bias_layer_1_filename, "bias_layer_1.txt");

        bias_layer_3_filename = (char *)malloc(strlen("bias_layer_3.txt")+1);
        strcpy(bias_layer_3_filename, "bias_layer_3.txt");

        weight_layer_1_filename = (char *)malloc(strlen("weight_layer_1.txt")+1);
        strcpy(weight_layer_1_filename, "weight_layer_1.txt");

        weight_layer_3_filename = (char *)malloc(strlen("weight_layer_3.txt")+1);
        strcpy(weight_layer_3_filename, "weight_layer_3.txt");
    } else if (mode == 2) { // Mode 2: testing
        bias_layer_1_filename = (char *)malloc(strlen("bias_trained_1.txt")+1);
        strcpy(bias_layer_1_filename, "bias_trained_1.txt");

        bias_layer_3_filename = (char *)malloc(strlen("bias_trained_3.txt")+1);
        strcpy(bias_layer_3_filename, "bias_trained_3.txt");

        weight_layer_1_filename = (char *)malloc(strlen("weight_trained_1.txt")+1);
        strcpy(weight_layer_1_filename, "weight_trained_1.txt");

        weight_layer_3_filename = (char *)malloc(strlen("weight_trained_3.txt")+1);
        strcpy(weight_layer_3_filename, "weight_trained_3.txt");
    } else if (mode == 3) { // Mode 3: demo / run
        if (argc < 3) {
            printf("./file <mode> [filename]");
            exit(-1);
        }

        input_filename = argv[2];

        bias_layer_1_filename = (char *)malloc(strlen("bias_trained_1.txt")+1);
        strcpy(bias_layer_1_filename, "bias_trained_1.txt");

        bias_layer_3_filename = (char *)malloc(strlen("bias_trained_3.txt")+1);
        strcpy(bias_layer_3_filename, "bias_trained_3.txt");

        weight_layer_1_filename = (char *)malloc(strlen("weight_trained_1.txt")+1);
        strcpy(weight_layer_1_filename, "weight_trained_1.txt");

        weight_layer_3_filename = (char *)malloc(strlen("weight_trained_3.txt")+1);
        strcpy(weight_layer_3_filename, "weight_trained_3.txt");
    }

    // layer input is initial pixels for 1 image
    Neuron layer_0[num_inputs] = {0};

    // An array of doubles (neurons)
    Neuron layer_1[num_neurons_per_layer] = {0};
    // Layer final is just our probability distribution, with 10 options
    Neuron layer_final[num_outputs] = {0};

    double inputs_layer_1[num_inputs] = {0};
    double inputs_layer_final[num_neurons_per_layer] = {0};

    load_bias_from_file_to_neurons(layer_1, bias_layer_1_filename);
    load_bias_from_file_to_neurons(layer_final, bias_layer_3_filename);
    load_weights_from_file_to_neurons(layer_1, weight_layer_1_filename, num_inputs, true);
    load_weights_from_file_to_neurons(layer_final, weight_layer_3_filename, num_neurons_per_layer, false);
    if (mode == 1 || mode == 2) { // We dont need mnist if we are demo running
        load_mnist();
    }
    // amount of training images (total bytes / size of one  image)
    size_t images_count;
    // This is a sort of fake 2d array - it will store all of the pixels of all of the images, but without absraction.
    double *image_list;
    int *true_labels;
    switch (mode) {
        case 1:
        // Training off of mnist
            images_count = sizeof(train_image) / sizeof(train_image[0]);
            image_list = (double*)calloc(images_count*num_inputs, sizeof(double));
            memcpy(image_list, train_image, images_count * num_inputs * sizeof(double));

            true_labels = (int*)calloc(images_count, sizeof(int));
            memcpy(true_labels, train_label, sizeof(int) * images_count);

            break;
        case 2:
        // Testing on mnist
            images_count = sizeof(test_image) / sizeof(test_image[0]);
            image_list = (double*)calloc(images_count*num_inputs, sizeof(double));
            memcpy(image_list, test_image, images_count * num_inputs * sizeof(double));

            true_labels = (int*)calloc(images_count, sizeof(int));
            memcpy(true_labels, test_label, sizeof(int) * images_count);

            break;
        case 3: { // These are necessary because the compiler was mad that i wasnt following standards for label declaration-definition
            // Testing on independent image
            images_count = 1;
            FILE *fptr = fopen(input_filename, "r");
            if (fptr == NULL) {
                perror("File Open Error Loading Biases");
                return -1;
            }
            double temp_double;
            int counter = 0;
            image_list = calloc(num_inputs, sizeof(double));
            while (counter < num_inputs && fscanf(fptr, "%lf", &temp_double) == 1) {
                image_list[counter++] = temp_double;
            }
            break;
        }

        default:
            printf("No maching switch case for mode");
            exit(-1);
            break;
    }

    double total_loss = 0;

    int right = 0;
    int wrong = 0;
    // For each image
    for (int i = 0; i < images_count; i+= 1) {
        // Size of an image (total bytes of image / size of one pixel(double))
        size_t length_of_image = num_inputs;

        // For each pixel:
        for (int j = 0; j < length_of_image; j++) {
            // Layer input element is the pixel
            // image_list is a fake 2d array
            layer_0[j].input_value = image_list[i * num_inputs + j];
            inputs_layer_1[j] = image_list[i * num_inputs + j];
        }
        // Now, layer input is loaded with the image

        // Populate layer 1
        for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
            // Convert to GSL vectors
            gsl_vector_view v_input = gsl_vector_view_array(inputs_layer_1, num_inputs);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_1[next_neuron].weights1, num_inputs);
            
            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer input
            dot_product += layer_1[next_neuron].bias;

            layer_1[next_neuron].pre_a = dot_product;
            // Apply activation function
            dot_product = relu(dot_product);
            // Set layer 1 neuron
            layer_1[next_neuron].output = dot_product;
        }

        // Populate layer final
        for (int next_neuron = 0; next_neuron < num_outputs; next_neuron++) {
            // Convert to GSL vectors
            for (int input_index = 0; input_index < num_neurons_per_layer; input_index++) {
                inputs_layer_final[input_index] = layer_1[input_index].output;
            }
            gsl_vector_view v_input = gsl_vector_view_array(inputs_layer_final, num_neurons_per_layer);
            gsl_vector_view v_weights = gsl_vector_view_array(layer_final[next_neuron].weights, num_neurons_per_layer);
            // Take dot product
            double dot_product = 0.0;
            gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
            // Apply bias going out of layer 2
            dot_product += layer_final[next_neuron].bias;
            layer_final[next_neuron].pre_a = dot_product;
            // No relu here
            // Set layer final neuron
            layer_final[next_neuron].output = dot_product;
        }

        // Time to softmax
        log_fast("Softmaxing now : counting total\n");
        // 1: Find the maximum output value for numerical stability
        double max = -INFINITY;
        for (int i = 0; i < num_outputs; i++) {
            if (layer_final[i].output > max)
                max = layer_final[i].output;
        }

        // 2: Compute the sum of exponentials of the shifted outputs
        double total = 0;
        for (int i = 0; i < num_outputs; i++) {
            total += exp(layer_final[i].output - max);
        }

        // 3: Compute the softmax percentages (probabilities)
        int predicted_class = 0;
        for (int i = 0; i < num_outputs; i++) {
            layer_final[i].percent = exp(layer_final[i].output - max) / total;

            // Track the class with the highest probability
            if (layer_final[i].percent > layer_final[predicted_class].percent) {
                predicted_class = i;
                if (mode == 3)
                    exit(predicted_class);
            }
        }

        // !! EVERYTHING PAST HERE HAPPENS ONLY FOR TRAINING/TESTING !! //

        // 4: Compare with actual label and increment counters
        
        if (predicted_class == true_labels[i]) {
            right++;
        } else {
            wrong++;
        }
        
        // !! START WEIGHT UPDATE !! //

        // If we are in training  mode
        if (mode == 1) {

            // calculate loss for this go
            double epsilon = 1e-10;
            double loss = -log(layer_final[train_label[i]].percent + epsilon); // | || || |_
            // Epsilon prevents log(0)

            log_fast("Total loss this image: %lf\n", loss);

            double delta_output[num_outputs];
            double delta_hidden[num_neurons_per_layer] = {0.0};

            // Compute deltas for output layer & accumulate for hidden layer
            for (int o = 0; o < num_outputs; o++) {
                double predicted = layer_final[o].percent;
                double target = (o == train_label[i]) ? 1.0 : 0.0;
                double error = predicted - target;
                delta_output[o] = error;
                log_fast("[%d] Neuron Predicted: %lf, Neuron Target: %lf \n", o, predicted, target);

                // Update output layer weights and biases
                for (int h = 0; h < num_neurons_per_layer; h++) {
                    // Update rule: w -= lr * delta * a_input
                    double w_backup = layer_final[o].weights[h];
                    layer_final[o].weights[h] -= learning_rate * error * layer_1[h].output;
                    delta_hidden[h] += error * w_backup;                
                    log_fast("[%d] Decreased weight [%d] by %lf\n", o, h, learning_rate * error * layer_1[h].output);
                    // nan problems are too frequent :(
                    if (isnan(layer_final[o].weights1[h])) {
                        log_fast("NAN\n\n");
                        printf("New Weight: %lf\n", layer_final[o].weights[h]);
                        printf("Learning Rate: %lf : Error: %lf : This Weight Training Output: %lf\n", learning_rate, error, layer_1[h].output);
                        exit(-1);
                    }
                    // Accumulate for hidden layer: δ_hidden[h] += δ_output[o] * w[o][h]
                }

                layer_final[o].bias -= learning_rate * error;
            }

            // Now compute deltas for hidden layer and update input weights
            for (int h = 0; h < num_neurons_per_layer; h++) {
                // ReLU derivative
                double grad = relu_derivative(layer_1[h].pre_a);
                double delta = grad * delta_hidden[h];

                for (int inp = 0; inp < num_inputs; inp++) {
                    // Update hidden weights
                    layer_1[h].weights1[inp] -= learning_rate * delta * layer_0[inp].input_value;
                    log_fast("[%d] Decreased weight [%d] by %lf\n", h, inp, learning_rate * delta * layer_0[inp].input_value);
                    if (isnan(layer_1[h].weights1[inp])) {
                        log_fast("NAN\n\n");
                        printf("New Weight: %lf\n", layer_final[h].weights[inp]);
                        printf("Learning Rate: %lf : delta: %lf : This Weight Training Output: %lf\n", learning_rate, delta, layer_1[h].output);
                        exit(-1);
                    }
                }
                layer_1[h].bias -= learning_rate * delta;
            }
        }

        // Selectively log
        if (i % 1000 == 0) {
            printf("%d images done, %lf%% accuracy\n", i, (double)(100)*right/(wrong+right));
        }

        FILE* f = fopen("log.txt", "w");
        fwrite(log_buffer, 1, log_index, f);
        fclose(f);
        // Reset the log index / tracker, it will begin re-writing the log buffer
        log_index=0;
    }


    printf("Right: %d\nWrong:%d\n", right, wrong);

    if (mode == 1) {
        //  Initialize arrays for later. It is much easier to 
        //  iterate over an array and write each one of the elements 
        //  than it is to go over each neuron and take its weight vector.
        double weights_layer_1[num_inputs*num_neurons_per_layer];
        double weights_layer_final[num_neurons_per_layer*num_neurons_per_layer];

        double biases_layer_1[num_inputs];
        double biases_layer_final[num_neurons_per_layer];
        int counter = 0;
    
        for (int i = 0; i < num_neurons_per_layer; i++) {
            for (int j = 0; j < num_inputs; j++) {
                weights_layer_1[counter++] = layer_1[i].weights1[j];
            }
        }
    
        counter = 0;
        for (int i = 0; i < num_neurons_per_layer; i++) {
            for (int j = 0; j < num_neurons_per_layer; j++) {
                weights_layer_final[counter++] = layer_final[i].weights[j];
            }
        }

        for (int i = 0; i < num_inputs; i++) {
            biases_layer_1[i] = layer_1[i].bias;
        }
    
        write_weights(weights_layer_1, num_inputs*num_neurons_per_layer, 1);
    
        write_weights(weights_layer_final, num_neurons_per_layer*num_neurons_per_layer, 3);
    }
}
/* 
    TODO:
        - modularize each layer step? 
*/