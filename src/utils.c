#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

size_t FindIndex( const int a[], size_t size, int value )
{
    size_t index = 0;

    while ( index < size && a[index] != value ) ++index;

    return ( index == size ? -1 : index );
}

void write_accuracy_to_file(double accuracy[training_image_count_thousands]) {
  FILE *fptr = fopen("accuracy.txt", "w");
  int counter = 0;
  while (counter < training_image_count_thousands) {
    fprintf(fptr, "%lf ", accuracy[counter++]);
  }
  printf("Write accuracy to file\n");
}

void write_weights(double *weights, size_t weights_len, int layer) {
  // Build the filename:

  // Get the layer int as a string
  int length = snprintf(NULL, 0, "%d", layer);
  char* layer_string = malloc(length+1);
  snprintf(layer_string, length+1, "%d", layer);

  // Build the actual filename string
  char prefix[] = "weight_trained_";
  char suffix[] = ".txt";
  char* filename = malloc(strlen(prefix) + length + strlen(suffix) + 1);
  strcpy(filename, prefix);
  strcat(filename, layer_string);
  strcat(filename, suffix);
  printf("%s\n", filename);

  char** result = malloc(sizeof(char));
  buildfilepath(result, filename, 2);
  if (remove(*result) != 0) {
    perror("Error removing the current weight file");
  }
  FILE *fptr;
  fptr = fopen(*result, "a");
  int counter = 0;
  for (size_t i = 0; i<weights_len; i++) {
    fprintf(fptr, "%lf ", weights[i]);
    counter++;
  }
  printf("Wrote %d weights to %s\n", counter, filename);

  fclose(fptr);
  free(result);
  free(filename);
  free(layer_string);
}

void write_bias(double* bias, size_t size, int layer) {
  // Build the filename:

  // Get the layer int as a string
  int length = snprintf(NULL, 0, "%d", layer);
  char* layer_string = malloc(length+1);
  snprintf(layer_string, length+1, "%d", layer);

  // Build the actual filename string
  char prefix[] = "bias_trained_";
  char suffix[] = ".txt";
  char* filename = malloc(strlen(prefix) + length + strlen(suffix) + 1);
  strcpy(filename, prefix);
  strcat(filename, layer_string);
  strcat(filename, suffix);
  printf("%s\n", filename);

  char** result = malloc(sizeof(char));
  buildfilepath(result, filename, 3);
  FILE *fptr;


  fptr = fopen(*result, "w");
  for (size_t i = 0; i<size; i++) {
    fprintf(fptr, "%lf ", bias[i]);
  }

  fclose(fptr);
  free(result);
  free(filename);
  free(layer_string);
}

void buildfilepath(char** result, char filename[], int type) {
  char* prefix_path;
  switch (type) {
    case 1:
      prefix_path = data_location;
      break;
    case 2:
      prefix_path = weights_location;
      break;
    case 3:
      prefix_path = bias_location;
      break;
    default:
      prefix_path = data_location;
      break;
  }
    // Allocate space for both strings and \0
  size_t total_len = strlen(prefix_path) + strlen(filename) + 1;
  // Temp string for storage
  char* temp = (char*)malloc(total_len);
  if (!temp) {
      perror("malloc failed");
      exit(1);
  }
  // Copy first + second part of path/filename to temp string
  strcpy(temp, prefix_path);
  strcat(temp, filename);

  // Set the result to the address of the temp string (pointer to [pointer to data]), setting the []
  *result = temp;
}

double relu(double x) {
  return MAX(0, x);
}

double relu_derivative(double x) {
  return x > 0 ? 1.0 : 0.0;
}

int load_weights_from_file_to_neurons(Neuron *neurons, char filename[], int size, bool input){
  // Build path
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename, 2);

  FILE *fptr = fopen(*filepath, "r");
  if (fptr == NULL) {
      perror("File Open Error Loading Weights");
      return -1;
  }

  int neuron_counter = 0;
  int neuron_index = 0;
  double temp_number;
  int number_loaded = 0;

  while(fscanf(fptr, "%lf", &temp_number) == 1) {
    if (input) {
    // weights1 is larger - since in the first layer, we need to store enough weights for num_inputs inputs
      neurons[neuron_counter].weights1[neuron_index] = temp_number;
    } else {
      neurons[neuron_counter].weights[neuron_index] = temp_number;
    }
    number_loaded++;

    neuron_index++;
    if (neuron_index >= size) {
      neuron_counter++;
      neuron_index = 0;
    }
  }
  if (number_loaded != num_neurons_per_layer*size) {
    printf("Error: weight mismatch. Expected: %d Loaded: %d\n", num_neurons_per_layer*size, number_loaded);
    exit(-1);
  }
  printf("Loaded %d / %d weights from %s\n", number_loaded, size*num_neurons_per_layer, filename);
  fclose(fptr);
  free(filepath);
  return 0;
}

int load_bias_from_file_to_neurons(Neuron *neurons, char filename[]) {
    // Build path
    char** filepath =  malloc(sizeof(char*));

    buildfilepath(filepath, filename, 3);

    FILE *fptr = fopen(*filepath, "r");
    if (fptr == NULL) {
        perror("File Open Error Loading Biases");
        return -1;
    }
  
    int bias_index = 0;
    double temp_number;
  
    while(fscanf(fptr, "%lf", &temp_number) == 1) {
      neurons[bias_index].bias = temp_number;
      bias_index++;
    }

    if (bias_index != num_neurons_per_layer) {
      printf("Error: bias mismatch. Expected: %d Loaded: %d\n", num_neurons_per_layer, bias_index);
      exit(-1);
    }
  
    printf("Loaded %d / %d biases from %s\n", bias_index, num_neurons_per_layer, filename);
    fclose(fptr);
    free(filepath);
    return 0;
}