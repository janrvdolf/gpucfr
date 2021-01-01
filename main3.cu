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

typedef struct node_t {
    struct node_t *parent;

    int player;
    INFORMATION_SET *information_set;

    // children
    int childs_count;
    struct node_t **childs;
} NODE;


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


    /* FREE MEMORY */
    free(information_set1);
    free(information_set2);

    return 0;
}