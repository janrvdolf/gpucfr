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


typedef float INFORMATION_SET;

void information_set_init(INFORMATION_SET* information_set, int number_of_actions) {
    int array_size = 4*number_of_actions + 1;

    for (int i = 0; i < array_size; i++) {
        information_set[i] = 0;
    }

    information_set[0] = number_of_actions;
}

void information_set_print(INFORMATION_SET* information_set) {
    int offset = 0;
    int number_of_actions = (int) information_set[0];

    printf("INFORMATION SET %p\n", information_set);
    printf("Number of actions %d\n", number_of_actions);
    printf("Current strategy:\n");
    offset += 1;
    for (int i = 0; i < number_of_actions; i++) {
        printf("%d: %f\n", i, information_set[i + offset]);
    }
    printf("Average strategy:\n");
    offset += number_of_actions;
    for (int i = 0; i < number_of_actions; i++) {
        printf("%d: %f\n", i, information_set[i + offset]);
    }
    printf("CF values:\n");
    offset += number_of_actions;
    for (int i = 0; i < number_of_actions; i++) {
        printf("%d: %f\n", i, information_set[i + offset]);
    }
    printf("Regrets:\n");
    offset += number_of_actions;
    for (int i = 0; i < number_of_actions; i++) {
        printf("%d: %f\n", i, information_set[i + offset]);
    }
}

typedef struct efg_node_t {
    struct efg_node_t *parent;

    int player;
    INFORMATION_SET *information_set;

    // children
    int childs_count;
    struct efg_node_t **childs;
} EFGNODE;


int main () {
    /* INFORMATION SETS */
    int number_of_actions = 3;
    // player 1
    size_t information_set1_size = 4 * number_of_actions * sizeof(float) + 1;
    INFORMATION_SET *information_set1 = (INFORMATION_SET*) malloc(information_set1_size);

    information_set_init(information_set1, number_of_actions);

    information_set_print(information_set1);
    // player 2
    size_t information_set2_size = 4 * number_of_actions * sizeof(float) + 1;
    INFORMATION_SET *information_set2 = (INFORMATION_SET*) malloc(information_set2_size);

    information_set_init(information_set2, number_of_actions);

    information_set_print(information_set2);

    /* INFORMATION SETS ON GPU */
    // player 1
    INFORMATION_SET *dev_information_set1 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_information_set1, information_set1_size)); // allocate on GPU
    CHECK_ERROR(cudaMemcpy(dev_information_set1, information_set1, information_set1_size, cudaMemcpyHostToDevice));  // copy data to GPU

    // player 2
    INFORMATION_SET *dev_information_set2 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_information_set2, information_set2_size)); // allocate on GPU
    CHECK_ERROR(cudaMemcpy(dev_information_set2, information_set2, information_set2_size, cudaMemcpyHostToDevice));  // copy data to GPU

    /* EFG NODES */
    // Player 1

    // node 1
    size_t efg_node_size = sizeof(EFGNODE);
    EFGNODE *node1 = (EFGNODE *) malloc(efg_node_size);
    if (!node1) {
        printf("Error malloc node1\n"); return 1;
    }
    node1->parent = NULL;
    node1->player = 1;
    node1->information_set = dev_information_set1;
    node1->childs_count = 0;
    node1->childs = NULL;
    // node 1 on GPU
    EFGNODE *dev_node1 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node1, efg_node_size)); // allocate on GPU

    // player 2

    // copy nodes to GPU memory
    CHECK_ERROR(cudaMemcpy(dev_node1, node1, efg_node_size, cudaMemcpyHostToDevice));

    /* FREE MEMORY */
    free(information_set1);
    free(information_set2);

    CHECK_ERROR(cudaFree(dev_information_set1));
    CHECK_ERROR(cudaFree(dev_information_set2));

    return 0;
}