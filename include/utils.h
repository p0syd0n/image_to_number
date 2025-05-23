
#include <stdbool.h>
#include <stddef.h>

#ifndef UTILS_H 
#define UTILS_H

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define data_location "data/"
#define weights_location "data/weight/"
#define bias_location "data/bias/"
#define num_inputs 784 // 28 * 28 pixels
#define num_outputs 10
#define num_neurons_per_layer 16

typedef struct {
  double input_value;
  double pre_a; // Pre-activation
  double output;
  double bias;
  double weights1[num_inputs];
  double weights[num_neurons_per_layer];
  double percent; // Used only in the last layer, as to not overwrite the output
  double delta;
} Neuron;

size_t FindIndex( const int a[], size_t size, int value );
void write_weights(double *weights, size_t weight_len, int layer);
void write_bias(double* bias, size_t size, int layer);
double relu(double x);

double relu_derivative(double x);

/*
   1: data 2: weights 3: bias
*/
void buildfilepath(char** result, char filename[], int type);

int load_bias_from_file_to_neurons(Neuron *neurons, char filename[]);

int load_weights_from_file_to_neurons(Neuron *neurons, char filename[], int size, bool input);

#endif