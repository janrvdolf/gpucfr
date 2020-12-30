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

#define BLOCK_DIMENSION 32


typedef struct efg_node_t {
    struct efg_node_t* parent;
    unsigned int player;
    float value;
    float reach_probability;
} EFG_NODE;


typedef struct node_t {
    int row;
    int column;
} NODE;


int main() {

    size_t node_mem_size = sizeof(NODE);
    NODE *node1 = (NODE*) malloc(node_mem_size);
    if (node1) {
        node1->row = 1;
        node1->column = 1;
    }  else {
        printf("Error: malloc node1\n");
        return 1;
    }
    NODE *node2 = (NODE*) malloc(node_mem_size);
    if (node2) {
        node2->row = 2;
        node2->column = 1;
    }  else {
        printf("Error: malloc node2\n");
        return 1;
    }
//    NODE *node3 = (NODE*) malloc(node_mem_size);
//    NODE *node4 = (NODE*) malloc(node_mem_size);


    return 0;
}
