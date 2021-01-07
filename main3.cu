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
    // init the number of actions
    information_set[0] = number_of_actions;

    int offset = 1;
    // init current_strategy
    for (int i = offset; i < number_of_actions + offset; i++) {
        information_set[i] = 1./number_of_actions;
    }

    offset += number_of_actions;
    // init average strategy
    for (int i = offset; i < number_of_actions + offset; i++) {
        information_set[i] = 0;
    }

    offset += number_of_actions;
    // init counterfactual values
    for (int i = offset; i < number_of_actions + offset; i++) {
        information_set[i] = 0;
    }

    offset += number_of_actions;
    // init regrets
    for (int i = offset; i < number_of_actions + offset; i++) {
        information_set[i] = 0;
    }

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
    float value;
    INFORMATION_SET *information_set; // TODO player should be probably in in INFORMATION_SET

    // children
    int childs_count;
    struct efg_node_t **childs;
} EFGNODE;

__global__ void cfv_kernel(EFGNODE ** terminal_nodes, int terminal_nodes_cnt) {
    int thread_id = threadIdx.x;

    if (thread_id < terminal_nodes_cnt) {
        if (thread_id == 1) {
            printf("terminal nodes cnt %d, first array element %p\n", terminal_nodes_cnt, terminal_nodes[thread_id]);

            EFGNODE *node = terminal_nodes[thread_id];

            float value = node->value;

            // here is terminal nodes, no information set

            printf("node %p, value %f\n", node, value);

            EFGNODE *from_node = node;
            node = node->parent; // a terminal node has always a parent node

            while (node) {
                printf("value %f, parent %p\n", value, node->parent);

                // search nodes's index in childs
                int child_idx = -1;
                for (int i = 0; i < node->childs_count; i++) {
                    EFGNODE **childs = node->childs;

                    if (from_node == childs[i]) {
                        child_idx = i;
                        break;
                    }
                }
                //printf("child's index is %d\n", child_idx);

                // TODO atomicly add to the cfv value in infoset

                if (child_idx > 0) {
                    // from current strategy get the action probability
                    INFORMATION_SET *information_set = node->information_set; // tohle bych mohl nacist do shared memory
                    printf("node information set value %f \n", information_set[0]); // zero index is for number of
                    int number_of_actions = information_set[0];

                    int offset = 1; // offset for strategy; // TODO refactor
                    float action_probability = information_set[offset + child_idx];
                    // multiply the value
                    value *= action_probability;

                    // atomic add to "information_set->cfv[action]"
                    offset = 1 + 2*number_of_actions;
                    atomicAdd(&information_set[offset + child_idx], value); // question if to do after or before previous line

                }

                from_node = node;
                node = node->parent;
            }
        }
    }
}


int main () {
    /* INFORMATION SETS */
    int number_of_actions = 3; // TODO max number of actions
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
    node1->value = 0.0;
    node1->information_set = dev_information_set1;
    node1->childs_count = 3;
    node1->childs = NULL;
    EFGNODE *dev_node1 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node1, efg_node_size));

    // Player 2
    // node 2
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node2 = (EFGNODE *) malloc(efg_node_size);
    if (!node2) {
        printf("Error malloc node2\n"); return 1;
    }
    node2->parent = dev_node1;
    node2->player = 2;
    node2->value = 0.0;
    node2->information_set = dev_information_set2;
    node2->childs_count = 3;
    node2->childs = NULL;
    EFGNODE *dev_node2 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node2, efg_node_size));

    // node 3
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node3 = (EFGNODE *) malloc(efg_node_size);
    if (!node3) {
        printf("Error malloc node3\n"); return 1;
    }
    node3->parent = dev_node1;
    node3->player = 2;
    node3->value = 0.0;
    node3->information_set = dev_information_set2;
    node3->childs_count = 3;
    node3->childs = NULL;
    EFGNODE *dev_node3 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node3, efg_node_size));

    // node 4
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node4 = (EFGNODE *) malloc(efg_node_size);
    if (!node4) {
        printf("Error malloc node4\n"); return 1;
    }
    node4->parent = dev_node1;
    node4->player = 2;
    node4->value = 0.0;
    node4->information_set = dev_information_set2;
    node4->childs_count = 3;
    node4->childs = NULL;
    EFGNODE *dev_node4 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node4, efg_node_size));

    // Terminal nodes
    // node 5
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node5 = (EFGNODE *) malloc(efg_node_size);
    if (!node5) {
        printf("Error malloc node5\n"); return 1;
    }
    node5->parent = dev_node2;
    node5->player = 0;
    node5->value = 0.0;
    node5->information_set = NULL;
    node5->childs_count = 0;
    node5->childs = NULL;
    EFGNODE *dev_node5 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node5, efg_node_size));
    // node 6
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node6 = (EFGNODE *) malloc(efg_node_size);
    if (!node6) {
        printf("Error malloc node6\n"); return 1;
    }
    node6->parent = dev_node2;
    node6->player = 0;
    node6->value = 1.0;
    node6->information_set = NULL;
    node6->childs_count = 0;
    node6->childs = NULL;
    EFGNODE *dev_node6 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node6, efg_node_size));
    // node 7
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node7 = (EFGNODE *) malloc(efg_node_size);
    if (!node7) {
        printf("Error malloc node7\n"); return 1;
    }
    node7->parent = dev_node2;
    node7->player = 0;
    node7->value = -1.0;
    node7->information_set = NULL;
    node7->childs_count = 0;
    node7->childs = NULL;
    EFGNODE *dev_node7 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node7, efg_node_size));
    // node 8
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node8 = (EFGNODE *) malloc(efg_node_size);
    if (!node8) {
        printf("Error malloc node8\n"); return 1;
    }
    node8->parent = dev_node3;
    node8->player = 0;
    node8->value = -1.0;
    node8->information_set = NULL;
    node8->childs_count = 0;
    node8->childs = NULL;
    EFGNODE *dev_node8 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node8, efg_node_size));
    // node 9
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node9 = (EFGNODE *) malloc(efg_node_size);
    if (!node9) {
        printf("Error malloc node9\n"); return 1;
    }
    node9->parent = dev_node3;
    node9->player = 0;
    node9->value = 0.0;
    node9->information_set = NULL;
    node9->childs_count = 0;
    node9->childs = NULL;
    EFGNODE *dev_node9 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node9, efg_node_size));
    // node 10
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node10 = (EFGNODE *) malloc(efg_node_size);
    if (!node10) {
        printf("Error malloc node10\n"); return 1;
    }
    node10->parent = dev_node3;
    node10->player = 0;
    node10->value = 1.0;
    node10->information_set = NULL;
    node10->childs_count = 0;
    node10->childs = NULL;
    EFGNODE *dev_node10 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node10, efg_node_size));
    // node 11
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node11 = (EFGNODE *) malloc(efg_node_size);
    if (!node11) {
        printf("Error malloc node11\n"); return 1;
    }
    node11->parent = dev_node4;
    node11->player = 0;
    node11->value = 1.0;
    node11->information_set = NULL;
    node11->childs_count = 0;
    node11->childs = NULL;
    EFGNODE *dev_node11 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node11, efg_node_size));
    // node 12
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node12 = (EFGNODE *) malloc(efg_node_size);
    if (!node12) {
        printf("Error malloc node12\n"); return 1;
    }
    node12->parent = dev_node4;
    node12->player = 0;
    node12->value = -1.0;
    node12->information_set = NULL;
    node12->childs_count = 0;
    node12->childs = NULL;
    EFGNODE *dev_node12 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node12, efg_node_size));
    // node 13
    efg_node_size = sizeof(EFGNODE);
    EFGNODE *node13 = (EFGNODE *) malloc(efg_node_size);
    if (!node13) {
        printf("Error malloc node13\n"); return 1;
    }
    node13->parent = dev_node4;
    node13->player = 0;
    node13->value = 0.0;
    node13->information_set = NULL;
    node13->childs_count = 0;
    node13->childs = NULL;
    EFGNODE *dev_node13 = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node13, efg_node_size));

    // Childs
    // node1
    size_t efgnode_childs_size = 3 * sizeof(EFGNODE**);
    EFGNODE **node1_childs = (EFGNODE**) malloc(efgnode_childs_size);
    node1_childs[0] = dev_node2;
    node1_childs[1] = dev_node3;
    node1_childs[2] = dev_node4;
    EFGNODE **dev_node1_childs = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node1_childs, efgnode_childs_size));
    CHECK_ERROR(cudaMemcpy(dev_node1_childs, node1_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
    node1->childs = dev_node1_childs;
    // TODO update childs_cnt here
    // node2
    efgnode_childs_size = 3 * sizeof(EFGNODE**);
    EFGNODE **node2_childs = (EFGNODE**) malloc(efgnode_childs_size);
    node2_childs[0] = dev_node5;
    node2_childs[1] = dev_node6;
    node2_childs[2] = dev_node7;
    EFGNODE **dev_node2_childs = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node2_childs, efgnode_childs_size));
    CHECK_ERROR(cudaMemcpy(dev_node2_childs, node2_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
    node2->childs = dev_node2_childs;
    // node3
    efgnode_childs_size = 3 * sizeof(EFGNODE**);
    EFGNODE **node3_childs = (EFGNODE**) malloc(efgnode_childs_size);
    node3_childs[0] = dev_node8;
    node3_childs[1] = dev_node9;
    node3_childs[2] = dev_node10;
    EFGNODE **dev_node3_childs = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node3_childs, efgnode_childs_size));
    CHECK_ERROR(cudaMemcpy(dev_node3_childs, node3_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
    node3->childs = dev_node3_childs;
    // node4
    efgnode_childs_size = 3 * sizeof(EFGNODE**);
    EFGNODE **node4_childs = (EFGNODE**) malloc(efgnode_childs_size);
    node4_childs[0] = dev_node11;
    node4_childs[1] = dev_node12;
    node4_childs[2] = dev_node13;
    EFGNODE **dev_node4_childs = NULL;
    CHECK_ERROR(cudaMalloc((void **) &dev_node4_childs, efgnode_childs_size));
    CHECK_ERROR(cudaMemcpy(dev_node4_childs, node4_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
    node4->childs = dev_node4_childs;

    // copy nodes to GPU memory
    CHECK_ERROR(cudaMemcpy(dev_node1, node1, efg_node_size, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(dev_node2, node2, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node3, node3, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node4, node4, efg_node_size, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(dev_node5, node5, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node6, node6, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node7, node7, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node8, node8, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node9, node9, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node10, node10, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node11, node11, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node12, node12, efg_node_size, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(dev_node13, node13, efg_node_size, cudaMemcpyHostToDevice));

    // Prepare a list of terminal nodes
    int terminal_nodes_size = 9;
    size_t terminal_nodes_ptr_size = sizeof(EFGNODE**) * terminal_nodes_size;
    EFGNODE **terminal_nodes = (EFGNODE**) malloc(terminal_nodes_ptr_size);
    if (!terminal_nodes) {
        printf("error malloc terminal nodes\n");
        return 1;
    }
    terminal_nodes[0] = dev_node5;
    terminal_nodes[1] = dev_node6;
    terminal_nodes[2] = dev_node7;
    terminal_nodes[3] = dev_node8;
    terminal_nodes[4] = dev_node9;
    terminal_nodes[5] = dev_node10;
    terminal_nodes[6] = dev_node11;
    terminal_nodes[7] = dev_node12;
    terminal_nodes[8] = dev_node13;
    EFGNODE** dev_terminal_nodes = (EFGNODE**) malloc(terminal_nodes_ptr_size);
    CHECK_ERROR(cudaMalloc((void **) &dev_terminal_nodes, terminal_nodes_ptr_size));
    CHECK_ERROR(cudaMemcpy(dev_terminal_nodes, terminal_nodes, terminal_nodes_ptr_size, cudaMemcpyHostToDevice));

    // information sets lengths by depth
    size_t information_sets_lengths_size = 2*sizeof(int);
    int *information_sets_lengths = (int *) malloc(information_sets_lengths_size);
    if (!information_sets_lengths) {
        printf("Error malloc information_sets_lengths\n");
        return 1;
    }
    information_sets_lengths[0] = 1;
    information_sets_lengths[1] = 1;

    // kernel - counterfactual value computation
    cfv_kernel<<<1, 32>>>(dev_terminal_nodes, terminal_nodes_size);

    /* FREE MEMORY */
    free(information_set1);
    free(information_set2);

    free(node1);

    free(node2);
    free(node3);
    free(node4);

    free(node5);
    free(node6);
    free(node7);
    free(node8);
    free(node9);
    free(node10);
    free(node11);
    free(node12);
    free(node13);

    free(node1_childs);

    free(node2_childs);
    free(node3_childs);
    free(node4_childs);

    free(terminal_nodes);

    CHECK_ERROR(cudaFree(dev_information_set1));
    CHECK_ERROR(cudaFree(dev_information_set2));

    CHECK_ERROR(cudaFree(dev_node1));

    CHECK_ERROR(cudaFree(dev_node2));
    CHECK_ERROR(cudaFree(dev_node3));
    CHECK_ERROR(cudaFree(dev_node4));

    CHECK_ERROR(cudaFree(dev_node5));
    CHECK_ERROR(cudaFree(dev_node6));
    CHECK_ERROR(cudaFree(dev_node7));
    CHECK_ERROR(cudaFree(dev_node8));
    CHECK_ERROR(cudaFree(dev_node9));
    CHECK_ERROR(cudaFree(dev_node10));
    CHECK_ERROR(cudaFree(dev_node11));
    CHECK_ERROR(cudaFree(dev_node12));
    CHECK_ERROR(cudaFree(dev_node13));

    CHECK_ERROR(cudaFree(dev_node1_childs));
    CHECK_ERROR(cudaFree(dev_node2_childs));
    CHECK_ERROR(cudaFree(dev_node3_childs));

    CHECK_ERROR(cudaFree(dev_terminal_nodes));

    return 0;
}