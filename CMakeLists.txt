cmake_minimum_required(VERSION 3.10)
find_package(GSL REQUIRED)


project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

include_directories(include)

add_executable(NeuralNetwork src/main.c src/utils.c)


target_link_libraries(NeuralNetwork m gsl gslcblas)

