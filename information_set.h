#ifndef INFORMATION_SET_H_
#define INFORMATION_SET_H_
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Simple function to check for CUDA runtime errors.
static void handleCUDAError(
        cudaError_t error,		// error code
        const char *file,		// file within error was generated
        int line )			// line where error occurs
{
    if (error != cudaSuccess) {	// any error -> display error message and terminate application
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define THREADS_PER_BLOCK 32u // THREADS_PER_BLOCK

#define CHECK_ERROR( error ) ( handleCUDAError( error, __FILE__, __LINE__ ) )


typedef float INFORMATION_SET;

class InformationSet {
private:
    size_t hash_t_ = 0;
    unsigned int number_of_actions_ = 0;

    size_t information_set_t_size_ = 0;
    INFORMATION_SET* information_set_t_ = NULL;

    INFORMATION_SET* dev_information_set_t_ = NULL;

    void init_();

public:
    InformationSet(size_t hash, unsigned number_of_actions);

    void memcpy_host_to_gpu ();

    void memcpy_gpu_to_host ();

    INFORMATION_SET* get_gpu_ptr();

    size_t get_hash();

    std::vector<double> get_current_strategy();

    std::vector<double> get_regrets();

    std::vector<double> get_cfv();

    float get_reach_probability ();

    std::vector<double> get_average_strategy();

    ~InformationSet();
};

#endif // INFORMATION_SET_H_