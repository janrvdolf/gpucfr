#include "gpucfr.h"

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

        infoset_data[1] = 0.0;
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
        atomicAdd(&information_set[1], reach_probability_i);
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

unsigned int GPUCFR::compute_blocks_number(int size) {
    unsigned int padded = std::ceil((float)size/THREADS_PER_BLOCK)*THREADS_PER_BLOCK;
    return padded/THREADS_PER_BLOCK;
}

GPUCFR::GPUCFR(std::string path) {
    path_ = path;

    load();
}

void GPUCFR::memcpy_host_to_gpu() {
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

void GPUCFR::memcpy_gpu_to_host () {
    // just information sets, because I need average strategy
    for (auto information_set: information_sets_) {
        information_set->memcpy_gpu_to_host();
    }
}

GPUCFR::~GPUCFR() {
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

void GPUCFR::load() {
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

void GPUCFR::run_iterations (unsigned int iterations)  {
    memcpy_host_to_gpu();

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

    memcpy_gpu_to_host();
};

float GPUCFR::elapsed_time() {
    return elapsed_time_;
}

void GPUCFR::run_iteration(float iteration) {
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

void GPUCFR::print_average_strategy() {
    for (auto information_set: information_sets_) {
        auto average_strategy = information_set->get_average_strategy();

        std::cout << "Information set hash " << information_set->get_hash() << std::endl;
        for (int i = 0; i < average_strategy.size(); i++) {
            std::cout << "\tAction index " << i << "; Action probability " << average_strategy.at(i) << std::endl;
        }
    }
}

