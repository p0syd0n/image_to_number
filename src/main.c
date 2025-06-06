#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <sys/types.h>
#include <stdarg.h>
#include "../include/mnist.h"
#include "../include/utils.h"

#define LOG_BUFFER_SIZE 102400000
#define CLIP_VALUE 100.0
#define EPSILON 1e-5
#define learning_rate 0.01

char log_buffer[LOG_BUFFER_SIZE];
size_t log_index = 0;
double accuracy_track_record[EPOCHS][training_image_count_thousands] = {0};

// Gradient Clipping Helper
double clip(double val, double limit) {
    return fmax(fmin(val, limit), -limit);
}

// Safety Check Helper
void safe_check(double val, const char* context) {
    if (!isfinite(val) || isnan(val)) {
        printf("Numerical error (%s): %lf\n", context, val);
        exit(1);
    }
}
// Buffer logging system
void log_fast(const char* format, ...) {
    return;
    va_list args;
    // va_start(args, format);
    // vprintf(format, args);  // Correct way to print with va_list
    // va_end(args);
    //return;
    // Or if you want to keep the buffer logging:
    va_start(args, format);
    log_index += vsnprintf(&log_buffer[log_index], LOG_BUFFER_SIZE - log_index, format, args);
    va_end(args);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("./%s <mode> [filename]", argv[0]);
        exit(-1);
    }

    // Strings for future filenames - they get decided based off of what mode we launch in
    char *weight_layer_1_filename;
    char *bias_layer_1_filename;

    char *weight_layer_2_filename;
    char *bias_layer_2_filename;

    char *weight_layer_3_filename;
    char *bias_layer_3_filename;

    char* input_filename;

    int mode = atoi(argv[1]);
    // Mode 1: training mode
    if (mode == 1) {
        weight_layer_1_filename = alloc_filename("weight_layer_1.txt");
        bias_layer_1_filename = alloc_filename("bias_layer_1.txt");
        weight_layer_2_filename = alloc_filename("weight_layer_2.txt");
        bias_layer_2_filename = alloc_filename("bias_layer_2.txt");
        weight_layer_3_filename = alloc_filename("weight_layer_3.txt");
        bias_layer_3_filename = alloc_filename("bias_layer_3.txt");

    } else if (mode == 2 || mode == 3) {
        if (mode == 3) {
            if (argc < 3) {
                printf("./%s <mode> [filename]\n", argv[0]);
                exit(-1);
            }
            input_filename = alloc_filename(argv[2]);
        }

        weight_layer_1_filename = alloc_filename("weight_trained_1.txt");
        bias_layer_1_filename = alloc_filename("bias_trained_1.txt");
        weight_layer_2_filename = alloc_filename("weight_trained_2.txt");
        bias_layer_2_filename = alloc_filename("bias_trained_2.txt");
        weight_layer_3_filename = alloc_filename("weight_trained_3.txt");
        bias_layer_3_filename = alloc_filename("bias_trained_3.txt");
    }


    printf("Mode: %d Input: %s\n", mode, input_filename);


    // layer input is initial pixels for 1 image
    InputNeuron layer_0[num_inputs] = {0};

    // An array of doubles (neurons)
    HiddenNeuron layer_1[num_neurons_per_layer] = {0};
    HiddenNeuron layer_2[num_neurons_per_layer] = {0};

    // Layer final is just our probability distribution, with 10 options
    OutputNeuron layer_final[num_outputs] = {0};

    double inputs_layer_1[num_inputs] = {0};
    double inputs_layer_2[num_neurons_per_layer] = {0};
    double inputs_layer_final[num_neurons_per_layer] = {0};

    load_weights_from_file_to_neurons_H(layer_1, weight_layer_1_filename, num_inputs, true);
    load_bias_from_file_to_neurons_H(layer_1, bias_layer_1_filename);

    load_weights_from_file_to_neurons_H(layer_2, weight_layer_2_filename, num_neurons_per_layer, false);
    load_bias_from_file_to_neurons_H(layer_2, bias_layer_2_filename);

    load_weights_from_file_to_neurons_O(layer_final, weight_layer_3_filename, num_neurons_per_layer, false);
    load_bias_from_file_to_neurons_O(layer_final, bias_layer_3_filename);
    
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
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // For each image
        for (int i = 0; i < images_count; i+= 1) {
            //printf("i(PRINTF):%d\n", i);
            log_fast("i:%d\n", i);
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
            log_fast("Populating layer 1\n");
            for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
                // Convert to GSL vectors
                gsl_vector_view v_input = gsl_vector_view_array(inputs_layer_1, num_inputs);
                gsl_vector_view v_weights = gsl_vector_view_array(layer_1[next_neuron].weights, num_inputs);
                
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
            //   log_fast("Layer 1 neuron %d output is %lf\n", next_neuron, dot_product);
            }
            
            // Populate layer 2
            log_fast("Populating layer 2\n");
            for (int next_neuron = 0; next_neuron < num_neurons_per_layer; next_neuron++) {
                for (int input_index = 0; input_index < num_neurons_per_layer; input_index++) {
                    inputs_layer_2[input_index] = layer_1[input_index].output;
                }
                // Convert to GSL vectors
                gsl_vector_view v_input = gsl_vector_view_array(inputs_layer_2, num_neurons_per_layer);
                gsl_vector_view v_weights = gsl_vector_view_array(layer_2[next_neuron].weights, num_neurons_per_layer);
                double sum = 0.0;
                for (int weight = 0; weight < num_neurons_per_layer; weight++) {
                    sum += layer_2[next_neuron].weights[weight];
                }
                // Take dot product
                double dot_product = 0.0;
                gsl_blas_ddot(&v_input.vector, &v_weights.vector, &dot_product);
                // Apply bias going out of layer input
                dot_product += layer_2[next_neuron].bias;

                layer_2[next_neuron].pre_a = dot_product;
                // Apply activation function
                dot_product = relu(dot_product);
                // Set layer 1 neuron
                layer_2[next_neuron].output = dot_product;
            //  log_fast("Layer 2 neuron %d output is %lf\n", next_neuron, dot_product);
            //  log_fast("The weights summed up to %lf\n", sum);
            }

            // Populate layer final
            log_fast("Populating layer 3\n");
            for (int next_neuron = 0; next_neuron < num_outputs; next_neuron++) {
                // Convert to GSL vectors
                for (int input_index = 0; input_index < num_neurons_per_layer; input_index++) {
                    inputs_layer_final[input_index] = layer_2[input_index].output;
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
            // log_fast("Layer final neuron %d output is %lf\n", next_neuron, dot_product);
            }

            // Time to softmax
            //log_fast("Softmaxing now : counting total\n");
            // 1: Find the maximum output value for numerical stability
            double max = -INFINITY;
            for (int q = 0; q < num_outputs; q++) {
                if (layer_final[q].output > max)
                    max = layer_final[q].output;
            }

            // 2: Compute the sum of exponentials of the shifted outputs
            double total = 0;
            for (int q = 0; q < num_outputs; q++) {
                total += exp(layer_final[q].output - max);
            }
            //log_fast("EXPONENT SUMMATION (SOFTMAX DENOMINATOR): %lf\n", total);

            // 3: Compute the softmax percentages (probabilities)
            int predicted_class = 0;
            for (int q = 0; q < num_outputs; q++) {
                layer_final[q].percent = exp(layer_final[q].output - max) / total;
            //  log_fast("Output neuron %d probabolity: %lf\n", q, layer_final[q].percent);
                if (isnan(layer_final[q].percent)) {
                    printf("NAN\n");
                    exit(-1);
                }
                log_fast("Neuron %d percent: %lf\n", q, layer_final[q].percent);

                // Track the class with the highest probability
                if (layer_final[q].percent > layer_final[predicted_class].percent) {
                    predicted_class = q;
                }
            }
            if (mode == 3)
                exit(predicted_class);

            // !! EVERYTHING PAST HERE HAPPENS ONLY FOR TRAINING/TESTING !! //

            // 4: Compare with actual label and increment counters
            int one_hot_vector[10] = {0};
            one_hot_vector[true_labels[i]] = 1;
            
            log_fast("Right answer: %d\n", true_labels[i]);
            log_fast("Model predicted: %d\n", predicted_class);
            log_fast("One Hot Vector:\n{");
            for (int k = 0; k < num_outputs; k++) {
                log_fast("%d, ", one_hot_vector[k]);
            }
            log_fast("}\n");

            if (predicted_class == true_labels[i]) {
                right++;
            } else {
                wrong++;
            }
            // Constants


            // RAH BACKPROPPPPPP
            if (mode == 1) {
                // | |i || |_ ehehehehehehehhhheheheeeheheheheee
                double safe_prob = fmax(layer_final[true_labels[i]].percent, EPSILON);
                double loss = -log(safe_prob);

                // Calculate blame
                for (int outputNeuron = 0; outputNeuron < num_outputs; outputNeuron++) {
                    layer_final[outputNeuron].blame = 0;
                    double error = layer_final[outputNeuron].percent - one_hot_vector[outputNeuron];
                    double gradient = error;
                    layer_final[outputNeuron].blame = clip(gradient, CLIP_VALUE);
                    layer_final[outputNeuron].percent = error;
                }

                for (int hiddenNeuron2 = 0; hiddenNeuron2 < num_neurons_per_layer; hiddenNeuron2++) {
                    layer_2[hiddenNeuron2].blame = 0;
                    for (int outputNeuron = 0; outputNeuron < num_outputs; outputNeuron++) {
                        layer_2[hiddenNeuron2].blame += layer_final[outputNeuron].blame * layer_final[outputNeuron].weights[hiddenNeuron2];
                    }
                    double gradient = layer_2[hiddenNeuron2].blame * relu_derivative(layer_2[hiddenNeuron2].pre_a);
                    layer_2[hiddenNeuron2].blame = clip(gradient, CLIP_VALUE);
                }


                for (int hiddenNeuron1 = 0; hiddenNeuron1 < num_neurons_per_layer; hiddenNeuron1++) {
                    layer_1[hiddenNeuron1].blame = 0;
                    for (int hiddenNeuron2 = 0; hiddenNeuron2 < num_neurons_per_layer; hiddenNeuron2++) {
                        layer_1[hiddenNeuron1].blame += layer_2[hiddenNeuron2].blame * layer_2[hiddenNeuron2].weights[hiddenNeuron1];
                    }
                    double gradient = layer_1[hiddenNeuron1].blame * relu_derivative(layer_1[hiddenNeuron1].pre_a);
                    layer_1[hiddenNeuron1].blame = clip(gradient, CLIP_VALUE);
                }


                // Update weights
                for (int outputNeuron = 0; outputNeuron < num_outputs; outputNeuron++) {
                    log_fast("[output] neuron %d\n", outputNeuron);
                    for (int weight = 0; weight < num_neurons_per_layer; weight++) {
                        double change = learning_rate * layer_final[outputNeuron].blame * layer_2[weight].output;
                        layer_final[outputNeuron].weights[weight] -= change;
                        log_fast("  weight %d updated by %lf ( %lf blame , %lf connecting neuron output )\n", weight, change, layer_final[outputNeuron].blame, layer_2[weight].output);
                    }
                    layer_final[outputNeuron].bias -= learning_rate * layer_final[outputNeuron].blame;
                }

                for (int hiddenNeuron2 = 0; hiddenNeuron2 < num_neurons_per_layer; hiddenNeuron2++) {
                    log_fast("[layer 2] neuron %d\n", hiddenNeuron2);
                    for (int weight = 0; weight < num_neurons_per_layer; weight++) {
                        double change = learning_rate * layer_2[hiddenNeuron2].blame * layer_1[weight].output;
                        layer_2[hiddenNeuron2].weights[weight] -= change;
                        log_fast("  weight %d updated by %lf ( %lf blame , %lf connecting neuron output )\n", weight, change, layer_2[hiddenNeuron2].blame, layer_1[weight].output);
                    }
                    layer_2[hiddenNeuron2].bias -= learning_rate * layer_2[hiddenNeuron2].blame;
                }

                for (int hiddenNeuron1 = 0; hiddenNeuron1 < num_neurons_per_layer; hiddenNeuron1++) {
                    log_fast("[layer 1] neuron %d\n", hiddenNeuron1);
                    for (int weight = 0; weight < num_inputs; weight++) {
                        double change = learning_rate * layer_1[hiddenNeuron1].blame * inputs_layer_1[weight];
                        layer_1[hiddenNeuron1].weights[weight] -= change;
                        log_fast("  weight %d updated by %lf ( %lf blame , %lf connecting neuron output ), \n", weight, change, layer_1[hiddenNeuron1].blame, inputs_layer_1[weight]);
                    }
                    layer_1[hiddenNeuron1].bias -= learning_rate * layer_1[hiddenNeuron1].blame;
                }

                // Selectively log
                if (i % 1000 == 0) {
                    printf("\r     epoch %d, %6d images done, %6.3lf%% accuracy     ", epoch, i, (double)(100)*right/(wrong+right));
                    fflush(stdout);
                    accuracy_track_record[epoch-1][i/1000] = (double)(100)*right/(wrong+right);
                }

                FILE* f = fopen("log.txt", "w");
                fwrite(log_buffer, 1, log_index, f);
                fclose(f);
                // Reset the log index / tracker, it will begin re-writing the log buffer
                log_index=0;
                //break;
            }
        }
    }
    


    printf("\n\nRight: %d\nWrong:%d\n", right, wrong);
    printf("%f %% right\n", 100*((float)right/(right+wrong)));

    if (mode == 1) {
        write_accuracy_to_file(accuracy_track_record);
        //  Initialize arrays for later. It is much easier to 
        //  iterate over an array and write each one of the elements 
        //  than it is to go over each neuron and take its weight vector.
        double weights_layer_1[num_inputs*num_neurons_per_layer];
        double biases_layer_1[num_neurons_per_layer];

        double weights_layer_2[num_neurons_per_layer*num_neurons_per_layer];
        double biases_layer_2[num_neurons_per_layer];

        double weights_layer_final[num_neurons_per_layer * num_outputs];
        double biases_layer_final[num_outputs];
        int counter = 0;
    
        for (int i = 0; i < num_neurons_per_layer; i++) {
            for (int j = 0; j < num_inputs; j++) {
                weights_layer_1[counter++] = layer_1[i].weights[j];
            }
        }
    
        counter = 0;
        for (int i = 0; i < num_outputs; i++) {
            for (int j = 0; j < num_neurons_per_layer; j++) {
                weights_layer_final[counter++] = layer_final[i].weights[j];
            }
        }


        counter = 0;
        for (int i = 0; i < num_neurons_per_layer; i++) {
            for (int j = 0; j < num_neurons_per_layer; j++) {
                weights_layer_2[counter++] = layer_2[i].weights[j];
            }
        }

        for (int i = 0; i < num_neurons_per_layer; i++) {
            biases_layer_1[i] = layer_1[i].bias;
        }

        for (int i = 0; i < num_neurons_per_layer; i++) {
            biases_layer_2[i] = layer_2[i].bias;
        }
        

        for (int i = 0; i < num_outputs; i++) {
            biases_layer_final[i] = layer_final[i].bias;
        }
    
        write_weights(weights_layer_1, num_inputs*num_neurons_per_layer, 1);
        write_bias(biases_layer_1, num_neurons_per_layer, 1);

        write_weights(weights_layer_2, num_neurons_per_layer*num_neurons_per_layer, 2);
        write_bias(biases_layer_2, num_neurons_per_layer, 2);

        write_weights(weights_layer_final, num_neurons_per_layer*num_neurons_per_layer, 3);
        write_bias(biases_layer_final, num_neurons_per_layer, 3);
    }
    free(weight_layer_1_filename);
    free(bias_layer_1_filename);

    free(weight_layer_2_filename);
    free(bias_layer_2_filename);

    free(bias_layer_3_filename);
    free(weight_layer_3_filename);
    free(image_list);
    
    if (mode == 1 || mode == 2)
        free(true_labels);

}
/* 
    TODO:
        - modularize each layer step? 
*/