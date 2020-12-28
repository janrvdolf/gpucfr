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
    float value;
    float reach_probability;
} EFG_NODE;


__global__ void rm_kernel(float* dev_infoset_data, unsigned int offset_current_strategy) {
    unsigned int thread_id = threadIdx.x;

    float number_of_actions = dev_infoset_data[0];

    if (thread_id == 0) {
        float sum = 0.0;
        for (int i = 0; i < (int) number_of_actions; i++) {
            sum += max(dev_infoset_data[i + offset_current_strategy], 0.0f);
        }
//
        if (sum > 0.0) {
            for (int i = 0; i < (int) number_of_actions; ++i) {
                dev_infoset_data[i + offset_current_strategy] = max(dev_infoset_data[i + offset_current_strategy], 0.0f)/sum;

                // TODO add to average strategy
            }

        } else {
            for (int i = 0; i < (int) number_of_actions; i++) {
                dev_infoset_data[i + offset_current_strategy] = 1.0f/number_of_actions;

                // TODO add to average strategy
            }
        }

    }
}

__global__ void pokus_reach_probabilities(EFG_NODE ** nodes) {
    int thread_id = threadIdx.x;

    if (thread_id == 0) {
        printf("Hello reach probs\n");
        EFG_NODE* node = nodes[1];


        printf("node %p\n", node);

        while (node) {
            printf("%f", node->value);

            node = node->parent;
        }
    }
}

__global__ void pokus_kernel(int* pokus_pole) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int rowAddr = row * 32;
    int threadID = rowAddr + col;

    pokus_pole[threadID] = col;
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

    // alocating 2D array 32*32
    int* dev_in_dvaDarr = NULL;

    int base_size = 32;
    size_t dvaD_array_memory_size = base_size*base_size*sizeof(int);

    int* dvaD_array = (int *) malloc(dvaD_array_memory_size);

    //init
    for (int i = 0; i < base_size*base_size; i++) {
        dvaD_array[i] = 0;
    }

    CHECK_ERROR(cudaMalloc((void **) &dev_in_dvaDarr, dvaD_array_memory_size));

    CHECK_ERROR(cudaMemcpy(dev_in_dvaDarr, dvaD_array, dvaD_array_memory_size, cudaMemcpyHostToDevice));

    dim3 blocks(32/BLOCK_DIMENSION, 32/BLOCK_DIMENSION);
    dim3 threads(BLOCK_DIMENSION, BLOCK_DIMENSION);

    pokus_kernel<<<blocks, threads>>>(dev_in_dvaDarr);

    CHECK_ERROR(cudaMemcpy(dvaD_array, dev_in_dvaDarr, dvaD_array_memory_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < base_size*base_size; i++) {
        std::cout << dvaD_array[i] << " ";

        if (i > 0 && (i + 1) % 32 == 0) {
            std::cout << std::endl;
        }
    }

    // build a tree

    EFG_NODE* root_node = (EFG_NODE*) malloc(sizeof(EFG_NODE));
    root_node->parent = NULL;
    root_node->value = 0.25;
    root_node->reach_probability = 0.0;

    EFG_NODE* child_node1 = (EFG_NODE*) malloc(sizeof(EFG_NODE));
//    child_node1->parent = root_node;
    child_node1->value = 0.25;
    child_node1->reach_probability = 0.0;

    EFG_NODE* child_node2 = (EFG_NODE*) malloc(sizeof(EFG_NODE));
//    child_node2->parent = root_node;
    child_node2->value = 0.25;
    child_node2->reach_probability = 0.0;

    size_t efg_nodes_memory_size = sizeof(EFG_NODE*) * 3;
    EFG_NODE** efg_nodes = (EFG_NODE**) malloc(efg_nodes_memory_size);


    EFG_NODE** dev_efg_nodes = NULL;

    size_t dev_efg_node_memory_size = sizeof(EFG_NODE);
    EFG_NODE *dev_root_node = NULL;
    EFG_NODE *dev_child_node1 = NULL;
    EFG_NODE *dev_child_node2 = NULL;



    CHECK_ERROR(cudaMalloc((void **) &dev_root_node, dev_efg_node_memory_size));
    CHECK_ERROR(cudaMalloc((void **) &dev_child_node1, dev_efg_node_memory_size));
    CHECK_ERROR(cudaMalloc((void **) &dev_child_node2, dev_efg_node_memory_size));

    efg_nodes[0] = dev_root_node;
    efg_nodes[1] = dev_child_node1;
    efg_nodes[2] = dev_child_node2;

    child_node1->parent = dev_root_node;
    printf("child_node1 %p\n", child_node1->parent);
    child_node2->parent = dev_root_node;
    printf("child_node2 %p\n", child_node2->parent);

    CHECK_ERROR(cudaMalloc((void **) &dev_efg_nodes, efg_nodes_memory_size));

    CHECK_ERROR(cudaMemcpy(dev_root_node, root_node, dev_efg_node_memory_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_child_node1, child_node1, dev_efg_node_memory_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_child_node2, child_node2, dev_efg_node_memory_size, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(dev_efg_nodes, efg_nodes, efg_nodes_memory_size, cudaMemcpyHostToDevice));

    pokus_reach_probabilities<<<1, 32>>>(dev_efg_nodes);
    // delete nodes
    CHECK_ERROR(cudaFree(dev_root_node));
    CHECK_ERROR(cudaFree(dev_child_node1));
    CHECK_ERROR(cudaFree(dev_child_node2));
    // delete array
    CHECK_ERROR(cudaFree(dev_efg_nodes));

    // delete nodes
    free(root_node);
    free(child_node1);
    free(child_node2);
    // delete array
    free(efg_nodes);

//    CHECK_ERROR(cudaMemcpy(host_infoset_data, dev_infoset_data, infoset_size_memory_size, cudaMemcpyDeviceToHost));

    /*
     Postavim strom na GPU u tak ze udelam tolik nodu, kolik je ve strome. Jeden node bude structura, ve ktere bude ukazatel na parent strukturu.
     Tyto struktury musi byt alokovany na gpu. Udelam spojeni EFG node -> Structura uzlu. Zavolam threads na kazdy uzel, while smycka zkonci v root stromu.
     Postavim ho tak, ze jak budu prochazet/stavet EFG strom tak se bude stavet i strom na GPU.

     Pocet infosetu v ramci uzlu uz mam zjisteni z diplomky.

     Vystavim pole hodnot pro current strategie atd. viz tabule/fotka v mobilu.

     Po zkonceni se zkopiruji je strategie. Jak to spustit jako 2D mrizku? Udelat is experiment - > vygeneruju 2D pole struktur s hodnotami column a sloupec
    a porovnam.
     * */
    unsigned int offset_current_strategy = 1;
    rm_kernel<<<1, 32>>>(dev_infoset_data, offset_current_strategy);

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
