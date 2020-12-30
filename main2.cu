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
    printf("Created node1 at %p\n", node1);
    if (node1) {
        node1->row = 1;
        node1->column = 1;
    }  else {
        printf("Error: malloc node1\n");
        return 1;
    }
    NODE *node2 = (NODE*) malloc(node_mem_size);
    printf("Created node2 at %p\n", node2);
    if (node2) {
        node2->row = 2;
        node2->column = 1;
    }  else {
        printf("Error: malloc node2\n");
        return 1;
    }
    NODE *node3 = (NODE*) malloc(node_mem_size);
    printf("Created node3 at %p\n", node3);
    if (node3) {
        node3->row = 2;
        node3->column = 1;
    }  else {
        printf("Error: malloc node2\n");
        return 1;
    }
//    NODE *node4 = (NODE*) malloc(node_mem_size);

    int node_levels_size = 2;
    int* node_levels = (int *) malloc(sizeof(int) * node_levels_size);
    node_levels[0] = 1;
    node_levels[1] = 2;


    NODE*** node_tree_array = (NODE***) malloc(sizeof(NODE***)*2);
    if (!node_tree_array) {
        printf("Error: node_tree_array malloc");
        return 1;
    }

    for (int i = 0; i < node_levels_size; i++) {
        printf("node_level %d\n", node_levels[i]);

        NODE **nodes_level_i = (NODE**) malloc(sizeof(NODE**)*node_levels[i]);

        if (i + 1 == 1) {
            nodes_level_i[0] = node1;
            printf("%p\n", nodes_level_i[0]);
        }

        if (i + 1 == 2) {
            nodes_level_i[0] = node2;
            nodes_level_i[1] = node3;
        }

        node_tree_array[i] = nodes_level_i;
    }

    // print
    for (int i = 0; i < node_levels_size; i++) {
        int node_level_size = node_levels[i];

        printf("node level %d, value %d\n", i, node_level_size);

        for (int j = 0; j < node_level_size; j++) {
            NODE *node = node_tree_array[i][j];
            printf("%p\n", node);
        }

    }

    return 0;
}
