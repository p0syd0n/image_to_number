#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../include/utils.h"

void buildfilepath(char** result, char filename[]) {
  // Allocate space for both strings and \0
  size_t total_len = strlen(data_location) + strlen(filename) + 1;
  // Temp string for storage
  char* temp = (char*)malloc(total_len);
  if (!temp) {
      perror("malloc failed");
      exit(1);
  }
  // Copy first + second part of path/filename to temp string
  strcpy(temp, data_location);
  strcat(temp, filename);

  // Set the result to the address of the temp string (pointer to [pointer to data]), setting the []
  *result = temp;
}


double relu(double x) {
  return MAX(0, x);
}

int load_weights_1_from_file(double array[num_neurons_per_layer][num_inputs], char filename[]){
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename);

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
  return 0;
}

int load_weights_other_from_file(double array[num_neurons_per_layer][num_neurons_per_layer], char filename[]){
  char** filepath =  malloc(sizeof(char*));
  buildfilepath(filepath, filename);
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
  return 0;
}
