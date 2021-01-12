#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>

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

typedef struct efg_node_t {
    struct efg_node_t *parent;

    int player;
    float reach_probability;
    float value;
    INFORMATION_SET *information_set; // TODO player should be probably in in INFORMATION_SET

    // children
    int childs_count;
    struct efg_node_t **childs;
} EFGNODE;

__global__ void rm_kernel(INFORMATION_SET ** dev_infoset_data, unsigned int information_set_size) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < information_set_size) {
        INFORMATION_SET * infoset_data = dev_infoset_data[thread_id];
        float number_of_actions_f = infoset_data[0];
        auto number_of_actions = (unsigned int) number_of_actions_f;
        unsigned int offset_current_strategy = 2;
        unsigned int offset_regrets = 2 + 3*number_of_actions;

        float sum = 0.0;
        for (int i = 0; i < number_of_actions; i++) {
            sum += max(infoset_data[i + offset_regrets], 0.0f);
        }
        // update the current strategy
        if (sum > 0.0) {
            for (int i = 0; i < number_of_actions; ++i) {
                infoset_data[i + offset_current_strategy] = max(infoset_data[i + offset_regrets], 0.0f)/sum;
            }

        } else {
            for (int i = 0; i < number_of_actions; i++) {
                infoset_data[i + offset_current_strategy] = 1.0f/number_of_actions_f;
            }
        }
    }
}

__global__ void average_strategy_kernel(INFORMATION_SET ** dev_infoset_data, unsigned int information_set_size, float iteration) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < information_set_size) {
        INFORMATION_SET * infoset_data = dev_infoset_data[thread_id];

        auto number_of_actions = (unsigned int) infoset_data[0];
        unsigned int offset_current_strategy = 2;
        unsigned int offset_average_strategy = 2 + number_of_actions;

        float reach_probability = infoset_data[1];

        for (int i = 0; i < number_of_actions; i++) {
            infoset_data[i + offset_average_strategy] = (iteration-1)/iteration * infoset_data[i + offset_average_strategy] + (1.0/iteration) * reach_probability * infoset_data[i + offset_current_strategy];
        }
    }
}

__global__ void rp_kernel(EFGNODE ** nodes, unsigned int nodes_size) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < nodes_size) {
        EFGNODE* node = nodes[thread_id];
        INFORMATION_SET *information_set = node->information_set;

        unsigned int node_player = node->player;
        float reach_probability_minus_i = 1.0;
        float reach_probability_i = 1.0;

        EFGNODE *from_node = node;
        EFGNODE *tmp = node->parent;
        while (tmp) {
            int child_idx = -1;
            for (int i = 0; i < tmp->childs_count; i++) {
                EFGNODE **children = tmp->childs;
                if (from_node == children[i]) {
                    child_idx = i;
                    break;
                }
            }
            if (child_idx > -1) {
                INFORMATION_SET *tmp_information_set = tmp->information_set;
                // get from the 'tmp' node's current strategy the action probability
                if (tmp->player != node_player) {
                    reach_probability_minus_i *= tmp_information_set[2+child_idx];
                } else {
                    reach_probability_i *= tmp_information_set[2+child_idx];
                }
            }
            from_node = tmp;
            tmp = tmp->parent;
        }

        node->reach_probability = reach_probability_minus_i;
        // updates the information set's reach probability
        information_set[1] = reach_probability_i;
    }
}

__global__ void cfv_kernel(EFGNODE ** terminal_nodes, unsigned int terminal_nodes_cnt) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < terminal_nodes_cnt) {
        EFGNODE *node = terminal_nodes[thread_id];

        float value = node->value;

        EFGNODE *from_node = node;
        node = node->parent; // a terminal node has always a parent node

        while (node) {
            // search nodes's index in childs
            int child_idx = -1;
            for (int i = 0; i < node->childs_count; i++) {
                EFGNODE **childs = node->childs;

                if (from_node == childs[i]) {
                    child_idx = i;
                    break;
                }
            }

            if (child_idx > -1) {
                INFORMATION_SET *information_set = node->information_set;
                int number_of_actions = information_set[0];
                int offset = 2; // offset for the current strategy

                float action_probability = information_set[offset + child_idx];

                offset = 2 + 2 * number_of_actions; // offset for counterfactual values

                float player_sigh = 1.0; // terminal values are stores for player one, so for player two multiply with -1
                if (node->player == 2) {
                    player_sigh = -1.0;
                }
                atomicAdd(&information_set[offset + child_idx], player_sigh * node->reach_probability * value);

                value *= action_probability;
            }
            from_node = node;
            node = node->parent;
        }
    }
}

__global__ void regret_update_kernel(INFORMATION_SET ** dev_infoset_data, unsigned int information_set_size) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < information_set_size) {
        INFORMATION_SET * infoset_data = dev_infoset_data[thread_id];
        auto number_of_actions = (unsigned int) infoset_data[0];
        unsigned int offset_current_strategy = 2;
        unsigned int offset_cfv_values = 2 + 2*number_of_actions;
        unsigned int offset_regrets = 2 + 3*number_of_actions;

        float expected_utility = 0;
        for (int i = 0; i < number_of_actions; i++) {
            expected_utility += infoset_data[i + offset_current_strategy] * infoset_data[i + offset_cfv_values];
        }

        for (int i = 0; i < number_of_actions; i++) {
            infoset_data[i + offset_regrets] += infoset_data[i + offset_cfv_values] - expected_utility;
        }
    }
}


class InformationSet {
private:
    size_t hash_t_ = 0;
    unsigned int number_of_actions_ = 0;

    size_t information_set_t_size_ = 0;
    INFORMATION_SET* information_set_t_ = NULL;

    INFORMATION_SET* dev_information_set_t_ = NULL;

    void init_(){
        // init the number of actions
        information_set_t_[0] = number_of_actions_;
        information_set_t_[1] = 0.0; // infoset reach probability
        unsigned int offset = 2;
        // init current_strategy
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 0;
        }
        offset += number_of_actions_;
        // init average strategy
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 0;
        }
        offset += number_of_actions_;
        // init counterfactual values
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 0;
        }
        offset += number_of_actions_;
        // init regrets
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 0;
        }
    }

public:
    InformationSet(size_t hash, unsigned number_of_actions) {
        hash_t_ = hash;
        number_of_actions_ = number_of_actions;

        information_set_t_size_ = (4 * number_of_actions_ + 2) * sizeof(INFORMATION_SET);
        information_set_t_ = (INFORMATION_SET*) malloc(information_set_t_size_);

        init_();

        CHECK_ERROR(cudaMalloc((void **) &dev_information_set_t_, information_set_t_size_));
    }

    void memcpy_host_to_gpu () {
        // copy data from CPU's RAM to GPU's global memory
        CHECK_ERROR(cudaMemcpy(dev_information_set_t_, information_set_t_, information_set_t_size_, cudaMemcpyHostToDevice));
    }

    void memcpy_gpu_to_host () {
        // copy data from GPU's global memory to CPU's RAM
        CHECK_ERROR(cudaMemcpy(information_set_t_, dev_information_set_t_, information_set_t_size_, cudaMemcpyDeviceToHost));
    }

    INFORMATION_SET* get_gpu_ptr() {
        return dev_information_set_t_;
    }

    size_t get_hash() {
        return hash_t_;
    }

    std::vector<double> get_current_strategy() {
        std::vector<double> returning_strategy;
        int offset = 2;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    std::vector<double> get_regrets() {
        std::vector<double> returning_strategy;
        int offset = 2+3*number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    std::vector<double> get_cfv() {
        std::vector<double> returning_strategy;
        int offset = 2+2*number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    float get_reach_probability () {
        return information_set_t_[1];
    }

    std::vector<double> get_average_strategy() {
        std::vector<double> returning_strategy;
        unsigned int offset = 2 + number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    ~InformationSet() {
        free(information_set_t_);
        information_set_t_ = NULL;

        CHECK_ERROR(cudaFree(dev_information_set_t_));
        dev_information_set_t_ = NULL;
    }
};


class Node {
private:
    Node *parent_ = nullptr;
    InformationSet *information_set_ = nullptr;

    EFGNODE *   node_t_     = NULL;
    EFGNODE *   dev_node_t_ = NULL;
    EFGNODE **  dev_children_ = NULL;

    // gtlib data:
    size_t   hash_ = 0;
    unsigned int    number_of_actions_ = 0;
    unsigned int    player_ = 0;
    size_t   parent_hash_ = 0;
    size_t   information_set_hash_ = 0;
    float           value_ = 0.0;
public:
    std::vector<Node*> children;

    Node(Node *parent, InformationSet *information_set) {
        parent_ = parent;
        information_set_ = information_set;

        size_t node_t_size = sizeof(EFGNODE);
        node_t_ = (EFGNODE*) malloc(node_t_size);

        CHECK_ERROR(cudaMalloc((void **) &dev_node_t_, node_t_size));
    }

    ~Node() {
        if (node_t_)
            free(node_t_);
        if (dev_node_t_)
            CHECK_ERROR(cudaFree(dev_node_t_));
        if (dev_children_)
            CHECK_ERROR(cudaFree(dev_children_));
    }

    EFGNODE* get_gpu_ptr() {
        return dev_node_t_;
    }

    void memcpy_host_to_gpu () {
            size_t node_t_size = sizeof(EFGNODE);

            if (parent_) {
                node_t_->parent = parent_->get_gpu_ptr();
            } else {
                node_t_->parent = NULL;
            }

            node_t_->player = player_;
            node_t_->reach_probability = 0.0;
            node_t_->value = value_;
            if (information_set_) {
                node_t_->information_set = information_set_->get_gpu_ptr();
            } else {
                node_t_->information_set = NULL;
            }
            node_t_->childs_count = children.size();
            // node's children
            size_t node_children_size = node_t_->childs_count * sizeof(EFGNODE**);
            EFGNODE **node_children = (EFGNODE**) malloc(node_children_size);
            for (int i = 0; i < children.size(); i++) {
                node_children[i] = children[i]->get_gpu_ptr();
            }
            CHECK_ERROR(cudaMalloc((void **) &dev_children_, node_children_size));
            CHECK_ERROR(cudaMemcpy(dev_children_, node_children, node_children_size, cudaMemcpyHostToDevice));
            node_t_->childs = dev_children_;
            // node to GPU
//            CHECK_ERROR(cudaMalloc((void **) &dev_node_t_, node_t_size));
            CHECK_ERROR(cudaMemcpy(dev_node_t_, node_t_, node_t_size, cudaMemcpyHostToDevice));
//        }
    }

    void update_gtlib_data(
            size_t   hash,
            unsigned int    number_of_actions,
            unsigned int    player,
            size_t   parent_hash,
            size_t   information_set_hash,
            float           value) {
        hash_ = hash;
        number_of_actions_ = number_of_actions;
        player_ = player;
        parent_hash_ = parent_hash;
        information_set_hash_ = information_set_hash;
        value_ = value;
    }

    void add_child(Node *child) {
        children.push_back(child);
    }

};


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

    unsigned int compute_blocks_number(int size) {
        unsigned int padded = std::ceil((float)size/THREADS_PER_BLOCK)*THREADS_PER_BLOCK;
        return padded/THREADS_PER_BLOCK;
    }

public:
    std::string path_;
    std::vector<std::vector<Node*>> game_tree_;

    GPUCFR(std::string path) {
        path_ = path;
    }

    void memcpy_host_to_gpu() {
        // information sets
        for (auto information_set: information_sets_) {
            information_set->memcpy_host_to_gpu();
        }
        // nodes
        for (const auto &nodes_vec: game_tree_) {
            for (auto node: nodes_vec) {
                node->memcpy_host_to_gpu();
            }
        }
        // terminal nodes
        unsigned int terminal_nodes_size = terminal_nodes_size_;
        size_t terminal_nodes_ptr_size = sizeof(EFGNODE**) * terminal_nodes_size;
        terminal_nodes_ = (EFGNODE**) malloc(terminal_nodes_ptr_size);
        unsigned int cnt = 0;
        for (auto node: game_tree_.at(game_tree_.size() - 1)) {
            terminal_nodes_[cnt++] = node->get_gpu_ptr();
        }
        CHECK_ERROR(cudaMalloc((void **) &dev_terminal_nodes_, terminal_nodes_ptr_size));
        CHECK_ERROR(cudaMemcpy(dev_terminal_nodes_, terminal_nodes_, terminal_nodes_ptr_size, cudaMemcpyHostToDevice));
        // information sets array
        size_t information_sets_size = sizeof(INFORMATION_SET**) * information_sets_.size();
        information_sets_t_ = (INFORMATION_SET**) malloc(information_sets_size);
        for (int i = 0; i < information_sets_.size(); i++) {
            information_sets_t_[i] = information_sets_.at(i)->get_gpu_ptr();
        }
        CHECK_ERROR(cudaMalloc((void **) &dev_informations_sets_, information_sets_size));
        CHECK_ERROR(cudaMemcpy(dev_informations_sets_, information_sets_t_, information_sets_size, cudaMemcpyHostToDevice));
        // nodes into an array
        size_t nodes_size = sizeof(EFGNODE**) * nodes_size_;
        nodes_ = (EFGNODE**) malloc(nodes_size);
        cnt = 0;
        for (const auto &nodes_per_depth: game_tree_) {
            for (auto node: nodes_per_depth) {
                nodes_[cnt++] = node->get_gpu_ptr();
            }
        }
        CHECK_ERROR(cudaMalloc((void **) &dev_nodes_, nodes_size));
        CHECK_ERROR(cudaMemcpy(dev_nodes_, nodes_, nodes_size, cudaMemcpyHostToDevice));
    }

    void memcpy_gpu_to_host () {
        // just information sets, because I need average strategy
        for (auto information_set: information_sets_) {
            information_set->memcpy_gpu_to_host();
        }


        for (auto information_set: information_sets_) {
            std::cout << "-- IS " << information_set->get_hash() << std::endl;

            std::cout << "Reach probability:" << std::endl;
            std::cout << information_set->get_reach_probability() << std::endl;

            std::vector<double> average_strategy = information_set->get_average_strategy();
            //std::cout << information_set->get_hash() << " - size " << strategy.size() << std::endl;


            std::cout << "Average strategy:" << std::endl;
            for (int j = 0; j < average_strategy.size(); j++) {
                std::cout << average_strategy[j] << " ";
            }
            std::cout << std::endl;

            std::vector<double> current_strategy = information_set->get_current_strategy();
            std::cout << "Current strategy:" << std::endl;
            for (int j = 0; j < current_strategy.size(); j++) {
                std::cout << current_strategy[j] << " ";
            }
            std::cout << std::endl;

            std::vector<double> regrets = information_set->get_regrets();
            std::cout << "Regrets:" << std::endl;
            for (int j = 0; j < regrets.size(); j++) {
                std::cout << regrets[j] << " ";
            }
            std::cout << std::endl;

            std::vector<double> cfv = information_set->get_cfv();
            std::cout << "CFV:" << std::endl;
            for (int j = 0; j < cfv.size(); j++) {
                std::cout << cfv[j] << " ";
            }

            std::cout << std::endl;
        }

    }

    ~GPUCFR() {
        // free nodes
        for (const auto &nodes_vec: game_tree_) {
            for (auto node: nodes_vec) {
                delete node;
            }
        }
        // free information sets
        for (InformationSet *information_set: information_sets_) {
            delete information_set;
        }

        free(terminal_nodes_);
        CHECK_ERROR(cudaFree(dev_terminal_nodes_));

        free(information_sets_t_);
        CHECK_ERROR(cudaFree(dev_informations_sets_));

        free(nodes_);
        CHECK_ERROR(cudaFree(dev_nodes_));
    }

    void load() {
        unsigned int max_depth = 0;

        std::ifstream input_file(path_);
        input_file >> max_depth;
        // loop through depths
        for (int i = 0; i < max_depth; i++) {
            unsigned int nodes_cnt = 0;
            std::vector<Node*> tmp_nodes_vec;
            input_file >> nodes_cnt;
            for (int j = 0; j < nodes_cnt; j++) {
                size_t node_hash = 0;
                unsigned int node_number_of_actions = 0;
                unsigned int node_player = 0;
                size_t node_parent_hash = 0;
                float node_value = 0.0;
                size_t information_set_hash = 0;

                input_file >> node_hash;
                input_file >> node_number_of_actions;
                input_file >> node_player;
                input_file >> node_parent_hash;
                input_file >> node_value;
                input_file >> information_set_hash;

                Node *node_parent = nullptr;
                if (node_parent_hash) {
                    auto result_node_parent = node_hash2node_ptr_.find(node_parent_hash);
                    if (result_node_parent != node_hash2node_ptr_.end()) {
                        node_parent = result_node_parent->second;
                    }
                }

                InformationSet *is = nullptr;
                if (information_set_hash) {
                    auto result_is = is_hash2is_ptr_.find(information_set_hash);
                    if (result_is != is_hash2is_ptr_.end()) {
                        is = result_is->second;
                    } else {
                        is = new InformationSet(information_set_hash, node_number_of_actions);

                        is_hash2is_ptr_.emplace(std::make_pair(information_set_hash, is));

                        information_sets_.push_back(is);
                    }
                }

                Node *node = new Node(node_parent, is);
                node->update_gtlib_data(node_hash, node_number_of_actions, node_player, node_parent_hash, information_set_hash, node_value);

                node_hash2node_ptr_.emplace(std::make_pair(node_hash, node));

                nodes_size_++;

                if (node_parent) {
                    node_parent->add_child(node);
                }

                tmp_nodes_vec.push_back(node);
            }
            game_tree_.push_back(tmp_nodes_vec);
        }
        input_file.close();

        terminal_nodes_size_ = game_tree_.at(game_tree_.size() - 1).size();
    }

    void run_iterations (int iterations)  {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        for (int i = 1; i < iterations; i++) {
            run_iteration((float ) i);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);


        cudaEventElapsedTime(&elapsed_time_, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };

    float elapsed_time() {
        return elapsed_time_;
    }

    void run_iteration(float iteration) {
        // Regret matching
        unsigned int data_size = information_sets_.size();
        int blocks = compute_blocks_number(data_size);
        rm_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_informations_sets_, data_size);
        cudaDeviceSynchronize();
        // Reach probabilities
        data_size = nodes_size_ - terminal_nodes_size_;
        blocks = compute_blocks_number(data_size);
        rp_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_nodes_, data_size);
        cudaDeviceSynchronize();
        // Average strategies
        data_size = information_sets_.size();
        blocks = compute_blocks_number(information_sets_.size());
        average_strategy_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_informations_sets_, data_size, iteration);
        cudaDeviceSynchronize();
        // Counterfactual values
        data_size = terminal_nodes_size_;
        blocks = compute_blocks_number(data_size);
        cfv_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_terminal_nodes_, data_size);
        cudaDeviceSynchronize();
        // Regrets
        data_size = information_sets_.size();
        blocks = compute_blocks_number(data_size);
        regret_update_kernel<<<blocks, THREADS_PER_BLOCK>>>(dev_informations_sets_, data_size);
        cudaDeviceSynchronize();
    }

    bool export_strategy_for_gtlib (std::string path) {
        // export average strategy
        std::ofstream output(path);
        output << information_sets_.size() << std::endl;
        for (auto information_set: information_sets_) {
            auto            average_strategy       = information_set->get_average_strategy();
            unsigned int    average_strategy_size  = average_strategy.size();
            output << information_set->get_hash()   << std::endl;
            output << average_strategy_size         << std::endl;
            for (int i = 0; i < average_strategy_size; i++) {
                output << average_strategy[i] << std::endl;
            }
        }
        output.close();
        return true;
    }

};







int main () {
    /* Goofspiel 2 */
    unsigned int iterations = 1000;
    GPUCFR instance1 = GPUCFR("../gs2.game");
    instance1.load();
    instance1.memcpy_host_to_gpu();
    instance1.run_iterations(iterations);
    instance1.memcpy_gpu_to_host();
    instance1.export_strategy_for_gtlib("gs2.strategy");
//    std::cout << "1000 iterations takes " << instance1.elapsed_time() << "ms on Goospiel 2" << std::endl;

//    iterations = 5000;
//    GPUCFR game_loader2 = GPUCFR("../gs2.game");
//    game_loader2.load();
//    game_loader2.memcpy_host_to_gpu();
//    game_loader2.run_iterations(iterations);
//    game_loader2.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader2.elapsed_time() << "ms on Goospiel 2" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader3 = GPUCFR("../gs2.game");
//    game_loader3.load();
//    game_loader3.memcpy_host_to_gpu();
//    game_loader3.run_iterations(iterations);
//    game_loader3.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader3.elapsed_time() << "ms on Goospiel 2" << std::endl;
//
//    /* Goofspiel 3 */
//    iterations = 1000;
//    GPUCFR game_loader4 = GPUCFR("../gs3.game");
//    game_loader4.load();
//    game_loader4.memcpy_host_to_gpu();
//    game_loader4.run_iterations(iterations);
//    game_loader4.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader4.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    iterations = 5000;
//    GPUCFR game_loader5 = GPUCFR("../gs3.game");
//    game_loader5.load();
//    game_loader5.memcpy_host_to_gpu();
//    game_loader5.run_iterations(iterations);
//    game_loader5.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader5.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader6 = GPUCFR("../gs3.game");
//    game_loader6.load();
//    game_loader6.memcpy_host_to_gpu();
//    game_loader6.run_iterations(iterations);
//    game_loader6.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader6.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    /* Goofspiel 3 */
//    iterations = 1000;
//    GPUCFR game_loader7 = GPUCFR("../gs4.game");
//    game_loader7.load();
//    game_loader7.memcpy_host_to_gpu();
//    game_loader7.run_iterations(iterations);
//    game_loader7.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader7.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
//    iterations = 5000;
//    GPUCFR game_loader8 = GPUCFR("../gs4.game");
//    game_loader8.load();
//    game_loader8.memcpy_host_to_gpu();
//    game_loader8.run_iterations(iterations);
//    game_loader8.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader8.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader9 = GPUCFR("../gs4.game");
//    game_loader9.load();
//    game_loader9.memcpy_host_to_gpu();
//    game_loader9.run_iterations(iterations);
//    game_loader9.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader9.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
    return 0;
}