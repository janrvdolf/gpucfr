cmake_minimum_required(VERSION 3.17)
project(gpucfr CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(gpucfr main.cu)

set_target_properties(
        gpucfr
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)