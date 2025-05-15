#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

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

int load_weights_1_from_file(double array[num_neurons_per_layer][num_inputs], char filename[]){
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename, 2);

  FILE *fptr = fopen(*filepath, "r");
  if (fptr == NULL) {
      perror("File Open Error Loading Weights");
      return -1;
  }

  double temp_number;
  int counter_row = 0;
  int counter_column = 0;

  while(fscanf(fptr, "%lf", &temp_number) == 1) {
    array[counter_row][counter_column] = temp_number;
    counter_column++;
    if (counter_column >= num_inputs) {
        counter_column = 0;
        counter_row ++;
    }
    if (counter_row >= num_neurons_per_layer) {
      break;
    }
    //printf("Read: %lf\n", temp_number);
  }
  printf("Loaded input weights from file\n");
  return 0;
}

int load_weights_other_from_file(double array[num_neurons_per_layer][num_neurons_per_layer], char filename[]){
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename, 2);
  FILE *fptr = fopen(*filepath, "r");
  if (fptr == NULL) {
      perror("File Open Error Loading Weights");
      return -1;
  }

  double temp_number;
  int counter_row = 0;
  int counter_column = 0;

  while(fscanf(fptr, "%lf", &temp_number) == 1) {
    array[counter_row][counter_column] = temp_number;
    counter_column++;
    if (counter_column >= num_neurons_per_layer) {
        counter_column = 0;
        counter_row ++;
    }
    if (counter_row >= num_neurons_per_layer) {
      break;
    }
    //printf("Read: %lf\n", temp_number);
  }
  printf("Loaded regular weights from file\n");
  return 0;
}


int load_weights_from_file_to_neurons(Neuron *neurons, char filename[], int size, bool input){
  // Build path
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename, 2);
  printf("Weight Filepath: %s\n", *filepath);

  FILE *fptr = fopen(*filepath, "r");
  if (fptr == NULL) {
      perror("File Open Error Loading Weights");
      return -1;
  }
  printf("Wasnt't a fopen issue\n");

  int neuron_counter = 0;
  int neuron_index = 0;
  double temp_number;
  int number_loaded = 0;


  while(fscanf(fptr, "%lf", &temp_number) == 1) {
    printf("[%d]: %lf\n", number_loaded, temp_number);
    //if (temp_number == 0) return -1;
    if (input) {
    // weights1 is larger - since in the first layer, we need to store enough weights for num_inputs inputs
      neurons[neuron_counter].weights1[neuron_index] = temp_number;
      printf("Set weight index %d of neuron %d to %lf\n", neuron_index, neuron_counter, temp_number);
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

  printf("Loaded %d weights from file to neurons\n", number_loaded);
  fclose(fptr);
  return 0;
}

int load_bias_from_file_to_neurons(Neuron *neurons, char filename[]) {
    // Build path
    char** filepath =  malloc(sizeof(char*));

    buildfilepath(filepath, filename, 3);
    printf("Bias Filepath: %s\n", *filepath);

    FILE *fptr = fopen(*filepath, "r");
    if (fptr == NULL) {
        perror("File Open Error Loading Biases");
        return -1;
    }
  
    int neuron_counter = 0;
    double temp_number;
  
    while(fscanf(fptr, "%lf", &temp_number) == 1) {
      neurons[neuron_counter].bias = temp_number;
      neuron_counter++;
    }
  
    printf("Loaded biases from file\n");
    fclose(fptr);
    return 0;
}