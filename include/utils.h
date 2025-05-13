

#ifndef UTILS_H 
#define UTILS_H

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define data_location "data/"
#define num_inputs 784 // 28 * 28 pixels
#define num_outputs 10
#define num_neurons_per_layer 16

typedef struct {
  double input;
  double pre_a; // Pre-activation
  double output;
  double bias;
  double *weights;
} Neuron;

double relu(double x);

// This one is special because there are more inputs, and therefore weights for the first layer of neurons
int load_weights_1_from_file(double array[num_neurons_per_layer][num_inputs], char filename[]);

int load_weights_other_from_file(double array[num_neurons_per_layer][num_neurons_per_layer], char filename[]);
// Just building a filepath from filename
void buildfilepath(char** result, char filename[]);

void initialize_bias(double* bias_array, int size, double value);
#endif