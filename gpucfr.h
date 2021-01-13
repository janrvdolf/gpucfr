#ifndef GPUCFR_H_
#define GPUCFR_H_
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>

#include "cuda_base.h"
#include "information_set.h"
#include "efg_node.h"

#define THREADS_PER_BLOCK 32u // THREADS_PER_BLOCK

/*!
 This class implements the Counterfactual regret minimization (CFR) algorithm in C++/C/CUDA for GPU. More specifically, the original Vanilla CFR by Martin Zinkevich.
 * */
class GPUCFR {
private:
    std::unordered_map<size_t, Node*> node_hash2node_ptr_;
    std::unordered_map<size_t, InformationSet*> is_hash2is_ptr_;

    std::vector<InformationSet*> information_sets_;

    EFGNODE **terminal_nodes_ = NULL;
    EFGNODE **dev_terminal_nodes_ = NULL;

    INFORMATION_SET **information_sets_t_ = NULL;
    INFORMATION_SET **dev_informations_sets_ = NULL;

    unsigned int nodes_size_ = 0;
    unsigned int terminal_nodes_size_ = 0;

    EFGNODE **nodes_ = NULL;
    EFGNODE **dev_nodes_ = NULL;

    float elapsed_time_ = 0.0;

    std::string path_;
    std::vector<std::vector<Node*>> game_tree_;

    unsigned int compute_blocks_number(int size);

    void memcpy_host_to_gpu();

    void memcpy_gpu_to_host ();

    void run_iteration(float iteration);

    void load();

public:
    //! Constructor for GPUCFR class
    /*!
     *
     * @param path Path to an EFG file "*.game" exported from GTLib
     */
    GPUCFR(std::string path);

    ~GPUCFR();

    void run_iterations (unsigned int iterations);

    float elapsed_time();

};

#endif // GPUCFR_H_