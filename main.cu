#include <iostream>

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

#define CHECK_ERROR( error ) ( handleCUDAError( error, __FILE__, __LINE__ ) )


__global__ void rm_kernel(float* dev_infoset_data, int strategy_offset) {
    unsigned int thread_id = threadIdx.x;

    float number_of_actions = dev_infoset_data[0];

    if (thread_id == 0) {
        float sum = 0.0;
        for (int i = 0; i < (int) number_of_actions; i++) {
            sum += max(dev_infoset_data[i + strategy_offset], 0.0f);
        }

        if (fabs(sum) <= 1e-8) {
            for (int i = 0; i < (int) number_of_actions; ++i) {
                dev_infoset_data[i + strategy_offset] = max(dev_infoset_data[i + strategy_offset], 0.0f)/sum;
            }

        } else {
            for (int i = 0; i < (int) number_of_actions; i++) {
                dev_infoset_data[i + strategy_offset] = 1.0f/number_of_actions;
            }
        }
    }
}

int main() {
    // GPU memory:
    float *dev_infoset_data = NULL;
    // CPU memory:
    float infoset_size = 3;
    float infoset_size_memory_units = 4*infoset_size + 1;

    auto host_infoset_data = (float *) malloc(sizeof(float) * infoset_size_memory_units);

    // prepare data
    for (int i = 0; i < infoset_size_memory_units; i++) {
        host_infoset_data[i] = 0;
    }
    host_infoset_data[0] = 3;

    // allocate data at GPU
    size_t infoset_size_memory_size = sizeof(float) * infoset_size_memory_units;
    CHECK_ERROR(cudaMalloc((void **) &dev_infoset_data, infoset_size_memory_size));
    // copy data to GPU
    CHECK_ERROR(cudaMemcpy(dev_infoset_data, host_infoset_data, infoset_size_memory_units, cudaMemcpyHostToDevice));

    /*
     Postavim strom na GPU u tak ze udelam tolik nodu, kolik je ve strome. Jeden node bude structura, ve ktere bude ukazatel na parent strukturu.
     Tyto struktury musi byt alokovany na gpu. Udelam spojeni EFG node -> Structura uzlu. Zavolam threads na kazdy uzel, while smycka zkonci v root stromu.
     Postavim ho tak, ze jak budu prochazet/stavet EFG strom tak se bude stavet i strom na GPU.

     Pocet infosetu v ramci uzlu uz mam zjisteni z diplomky.

     Vystavim pole hodnot pro current strategie atd. viz tabule/fotka v mobilu.

     Po zkonceni se zkopiruji je strategie. Jak to spustit jako 2D mrizku? Udelat is experiment - > vygeneruju 2D pole struktur s hodnotami column a sloupec
    a porovnam.
     * */
    int strategy_offset = 1;
    rm_kernel<<<1, 32>>>(dev_infoset_data, strategy_offset);

    // copy data from GPU to host
    CHECK_ERROR(cudaMemcpy(host_infoset_data, dev_infoset_data, infoset_size_memory_size, cudaMemcpyDeviceToHost));

    std::cout << "Output Data:" << std::endl;
    for (int i = 0; i < infoset_size_memory_units; i++) {
        std::cout << host_infoset_data[i] << std::endl;
    }

    // free allocated memory
    CHECK_ERROR(cudaFree(dev_infoset_data));

    free(host_infoset_data);

    return 0;
}
