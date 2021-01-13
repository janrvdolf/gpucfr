cmake_minimum_required(VERSION 3.17)
project(gpucfr CUDA)

set(CMAKE_CUDA_STANDARD 14)

#add_executable(gpucfr main.cu)
#
#set_target_properties(
#        gpucfr
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

#add_executable(gpucfr2 main2.cu)
#
#set_target_properties(
#        gpucfr2
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

include_directories(${PROJECT_SOURCE_DIR})

set (SOURCE_FILES
        information_set.cu
        )

add_executable(gpucfr3 main3.cu ${SOURCE_FILES})
set_target_properties(
        gpucfr3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)