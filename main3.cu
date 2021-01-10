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
    float reach_probability;
    float value;
    INFORMATION_SET *information_set; // TODO player should be probably in in INFORMATION_SET

    // children
    int childs_count;
    struct efg_node_t **childs;
} EFGNODE;

__global__ void rm_kernel(INFORMATION_SET ** dev_infoset_data, unsigned int information_set_size, float iteration) {
    unsigned int thread_id = threadIdx.x;

    if (thread_id == 0 && thread_id < information_set_size) {
        INFORMATION_SET * infoset_data = dev_infoset_data[thread_id];
        float number_of_actions_f = infoset_data[0];
        auto number_of_actions = (unsigned int) infoset_data[0];
        unsigned int offset_current_strategy = 1;
        unsigned int offset_average_strategy = 1 + number_of_actions;

        float sum = 0.0;
        for (int i = 0; i < number_of_actions; i++) {
            sum += max(infoset_data[i + offset_current_strategy], 0.0f);
        }
        // update the current strategy
        if (sum > 0.0) {
            for (int i = 0; i < number_of_actions; ++i) {
                infoset_data[i + offset_current_strategy] = max(infoset_data[i + offset_current_strategy], 0.0f)/sum;
            }

        } else {
            for (int i = 0; i < number_of_actions; i++) {
                infoset_data[i + offset_current_strategy] = 1.0f/number_of_actions_f;
            }
        }
        // update the average strategy ... asi rozdelit
        for (int i = 0; i < number_of_actions; i++) {
            // TODO chybi tu reach prob infosetu
            infoset_data[i + offset_average_strategy] = (iteration-1)/iteration * infoset_data[i + offset_average_strategy] + (1.0/iteration) * infoset_data[i + offset_current_strategy];
        }
    }
}

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
                printf("----\n");
                printf("value %f, parent %p, childs %d\n", value, node->parent, node->childs_count);

                // search nodes's index in childs
                int child_idx = -1;
                for (int i = 0; i < node->childs_count; i++) {
                    EFGNODE **childs = node->childs;
                    printf("child %p\n", childs[i]);

                    if (from_node == childs[i]) {
                        child_idx = i;
                        break;
                    }
                }
                printf("child's index is %d\n", child_idx);
                INFORMATION_SET *information_set = node->information_set; // tohle bych mohl nacist do shared memory
                int number_of_actions = information_set[0];

                if (child_idx > -1) {
                    // from current strategy get the action probability
                    int offset = 1; // offset for strategy; // TODO refactor
                    float action_probability = information_set[offset + child_idx];

                    printf("node information set value %f, act prob %f\n", information_set[0], action_probability); // zero index is for number of


                    // multiply the value
                    value *= action_probability;

                    // atomic add to "information_set->cfv[action]"
                    offset = 1 + 2*number_of_actions;
                    atomicAdd(&information_set[offset + child_idx], value); // question if to do after or before previous line

                }

                from_node = node;
                node = node->parent;
            }

            printf("final value %f", value);
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
        unsigned int offset = 1;
        // init current_strategy
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 1./number_of_actions_;
        }
        offset += number_of_actions_;
        // init average strategy
        for (unsigned int i = offset; i < number_of_actions_ + offset; i++) {
            information_set_t_[i] = 0; // TODO check if this should also have uniform strategy
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

        information_set_t_size_ = (4 * number_of_actions_ + 1) * sizeof(INFORMATION_SET);
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
        int offset = 1;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    std::vector<double> get_average_strategy() {
        std::vector<double> returning_strategy;
        unsigned int offset = 1 + number_of_actions_;
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
//        if (node_t_ == NULL && dev_node_t_ == NULL && dev_children_ == NULL) {
            size_t node_t_size = sizeof(EFGNODE);
//            node_t_ = (EFGNODE*) malloc(node_t_size);

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


class GameLoader {
private:
    std::unordered_map<size_t, Node*> node_hash2node_ptr_;
    std::unordered_map<size_t, InformationSet*> is_hash2is_ptr_;

    std::vector<InformationSet*> information_sets_;

    EFGNODE **terminal_nodes_ = NULL;
    EFGNODE **dev_terminal_nodes_ = NULL;

    INFORMATION_SET **information_sets_t_ = NULL;
    INFORMATION_SET **dev_informations_sets_ = NULL;

public:
    std::string path_;
    std::vector<std::vector<Node*>> game_tree_;

    GameLoader(std::string path) {
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
        int terminal_nodes_size = game_tree_.at(game_tree_.size() - 1).size();
        size_t terminal_nodes_ptr_size = sizeof(EFGNODE**) * terminal_nodes_size;
        terminal_nodes_ = (EFGNODE**) malloc(terminal_nodes_ptr_size);
        int cnt = 0;
        for (auto node: game_tree_.at(game_tree_.size() - 1)) {
            terminal_nodes_[cnt++] = node->get_gpu_ptr();
        }
        dev_terminal_nodes_ = (EFGNODE**) malloc(terminal_nodes_ptr_size);
        CHECK_ERROR(cudaMalloc((void **) &dev_terminal_nodes_, terminal_nodes_ptr_size));
        CHECK_ERROR(cudaMemcpy(dev_terminal_nodes_, terminal_nodes_, terminal_nodes_ptr_size, cudaMemcpyHostToDevice));
        // information sets array
        size_t information_sets_size = sizeof(INFORMATION_SET**) * information_sets_.size();
        information_sets_t_ = (INFORMATION_SET**) malloc(information_sets_size);
        for (int i = 0; i < information_sets_.size(); i++) {
            information_sets_t_[i] = information_sets_.at(i)->get_gpu_ptr();
        }
        dev_informations_sets_ = (INFORMATION_SET **) malloc(information_sets_size);
        CHECK_ERROR(cudaMalloc((void **) &dev_informations_sets_, information_sets_size));
        CHECK_ERROR(cudaMemcpy(dev_informations_sets_, information_sets_t_, information_sets_size, cudaMemcpyHostToDevice));
    }

    void memcpy_gpu_to_host () {
        // just information sets, because I need average strategy
        for (auto information_set: information_sets_) {
            information_set->memcpy_gpu_to_host();
        }

        std::cout << std::endl; // TODO remove

        for (auto information_set: information_sets_) {
            std::vector<double> strategy = information_set->get_current_strategy();
            std::cout << information_set->get_hash() << " - size " << strategy.size() << std::endl;
            for (int j = 0; j < strategy.size(); j++) {
                std::cout << strategy[j] << " ";
            }
            std::cout << std::endl;
        }

    }

    ~GameLoader() {
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

    }

    /*
    output_file << node_hash << std::endl;
    output_file << node_number_of_actions << std::endl;
    output_file << node_player << std::endl;
    output_file << node_parent_hash << std::endl;
    output_file << information_set_hash << std::endl;
    */

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

                if (node_parent) {
                    node_parent->add_child(node);
                }

                tmp_nodes_vec.push_back(node);
            }
            game_tree_.push_back(tmp_nodes_vec);
        }
        input_file.close();
    }

    void run_iteration(int iteration) {
        rm_kernel<<<1, 32>>>(dev_informations_sets_, information_sets_.size(), iteration);

        cfv_kernel<<<1, 32>>>(dev_terminal_nodes_, 4);
    }

    void print_nodes() {
        // free nodes
        int cnt = 0;
        for (const auto &nodes_vec: game_tree_) {
            std::cout << "Depth " << ++cnt << std::endl;
            for (Node *node: nodes_vec) {
                std::cout << "->\t Node " << node << std::endl;
                std::cout << "\tChilds:" << std::endl;
                for (Node *child: node->children) {
                    std::cout << child << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }

    bool export_strategy (std::string out_path) { // TODO output strategy from information sets hash -> float values
        return true;
    }

};







int main () {
    GameLoader game_loader = GameLoader("/home/ruda/CLionProjects/gpucfr/output.game");
    game_loader.load();
//    game_loader.print_nodes();
    game_loader.memcpy_host_to_gpu();
    for (int i = 1; i < 2; i++) {
        game_loader.run_iteration(i);
    }
    game_loader.memcpy_gpu_to_host();


    /*
     * Rock-Paper-Scissors
     * */

//    /* INFORMATION SETS */
//    int number_of_actions = 3; // TODO max number of actions
//    // player 1
//    size_t information_set1_size = 4 * number_of_actions * sizeof(float) + 1;
//    INFORMATION_SET *information_set1 = (INFORMATION_SET*) malloc(information_set1_size);
//
//    information_set_init(information_set1, number_of_actions);
//
//    information_set_print(information_set1);
//    // player 2
//    size_t information_set2_size = 4 * number_of_actions * sizeof(float) + 1;
//    INFORMATION_SET *information_set2 = (INFORMATION_SET*) malloc(information_set2_size);
//
//    information_set_init(information_set2, number_of_actions);
//
//    information_set_print(information_set2);
//
//    /* INFORMATION SETS ON GPU */
//    // player 1
//    INFORMATION_SET *dev_information_set1 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_information_set1, information_set1_size)); // allocate on GPU
//    CHECK_ERROR(cudaMemcpy(dev_information_set1, information_set1, information_set1_size, cudaMemcpyHostToDevice));  // copy data to GPU
//
//    // player 2
//    INFORMATION_SET *dev_information_set2 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_information_set2, information_set2_size)); // allocate on GPU
//    CHECK_ERROR(cudaMemcpy(dev_information_set2, information_set2, information_set2_size, cudaMemcpyHostToDevice));  // copy data to GPU
//
//    /* EFG NODES */
//    // Player 1
//
//    // node 1
//    size_t efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node1 = (EFGNODE *) malloc(efg_node_size);
//    if (!node1) {
//        printf("Error malloc node1\n"); return 1;
//    }
//    node1->parent = NULL;
//    node1->player = 1;
//    node1->value = 0.0;
//    node1->information_set = dev_information_set1;
//    node1->childs_count = 3;
//    node1->childs = NULL;
//    EFGNODE *dev_node1 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node1, efg_node_size));
//
//    // Player 2
//    // node 2
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node2 = (EFGNODE *) malloc(efg_node_size);
//    if (!node2) {
//        printf("Error malloc node2\n"); return 1;
//    }
//    node2->parent = dev_node1;
//    node2->player = 2;
//    node2->value = 0.0;
//    node2->information_set = dev_information_set2;
//    node2->childs_count = 3;
//    node2->childs = NULL;
//    EFGNODE *dev_node2 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node2, efg_node_size));
//
//    // node 3
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node3 = (EFGNODE *) malloc(efg_node_size);
//    if (!node3) {
//        printf("Error malloc node3\n"); return 1;
//    }
//    node3->parent = dev_node1;
//    node3->player = 2;
//    node3->value = 0.0;
//    node3->information_set = dev_information_set2;
//    node3->childs_count = 3;
//    node3->childs = NULL;
//    EFGNODE *dev_node3 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node3, efg_node_size));
//
//    // node 4
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node4 = (EFGNODE *) malloc(efg_node_size);
//    if (!node4) {
//        printf("Error malloc node4\n"); return 1;
//    }
//    node4->parent = dev_node1;
//    node4->player = 2;
//    node4->value = 0.0;
//    node4->information_set = dev_information_set2;
//    node4->childs_count = 3;
//    node4->childs = NULL;
//    EFGNODE *dev_node4 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node4, efg_node_size));
//
//    // Terminal nodes
//    // node 5
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node5 = (EFGNODE *) malloc(efg_node_size);
//    if (!node5) {
//        printf("Error malloc node5\n"); return 1;
//    }
//    node5->parent = dev_node2;
//    node5->player = 0;
//    node5->value = 0.0;
//    node5->information_set = NULL;
//    node5->childs_count = 0;
//    node5->childs = NULL;
//    EFGNODE *dev_node5 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node5, efg_node_size));
//    // node 6
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node6 = (EFGNODE *) malloc(efg_node_size);
//    if (!node6) {
//        printf("Error malloc node6\n"); return 1;
//    }
//    node6->parent = dev_node2;
//    node6->player = 0;
//    node6->value = 1.0;
//    node6->information_set = NULL;
//    node6->childs_count = 0;
//    node6->childs = NULL;
//    EFGNODE *dev_node6 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node6, efg_node_size));
//    // node 7
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node7 = (EFGNODE *) malloc(efg_node_size);
//    if (!node7) {
//        printf("Error malloc node7\n"); return 1;
//    }
//    node7->parent = dev_node2;
//    node7->player = 0;
//    node7->value = -1.0;
//    node7->information_set = NULL;
//    node7->childs_count = 0;
//    node7->childs = NULL;
//    EFGNODE *dev_node7 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node7, efg_node_size));
//    // node 8
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node8 = (EFGNODE *) malloc(efg_node_size);
//    if (!node8) {
//        printf("Error malloc node8\n"); return 1;
//    }
//    node8->parent = dev_node3;
//    node8->player = 0;
//    node8->value = -1.0;
//    node8->information_set = NULL;
//    node8->childs_count = 0;
//    node8->childs = NULL;
//    EFGNODE *dev_node8 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node8, efg_node_size));
//    // node 9
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node9 = (EFGNODE *) malloc(efg_node_size);
//    if (!node9) {
//        printf("Error malloc node9\n"); return 1;
//    }
//    node9->parent = dev_node3;
//    node9->player = 0;
//    node9->value = 0.0;
//    node9->information_set = NULL;
//    node9->childs_count = 0;
//    node9->childs = NULL;
//    EFGNODE *dev_node9 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node9, efg_node_size));
//    // node 10
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node10 = (EFGNODE *) malloc(efg_node_size);
//    if (!node10) {
//        printf("Error malloc node10\n"); return 1;
//    }
//    node10->parent = dev_node3;
//    node10->player = 0;
//    node10->value = 1.0;
//    node10->information_set = NULL;
//    node10->childs_count = 0;
//    node10->childs = NULL;
//    EFGNODE *dev_node10 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node10, efg_node_size));
//    // node 11
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node11 = (EFGNODE *) malloc(efg_node_size);
//    if (!node11) {
//        printf("Error malloc node11\n"); return 1;
//    }
//    node11->parent = dev_node4;
//    node11->player = 0;
//    node11->value = 1.0;
//    node11->information_set = NULL;
//    node11->childs_count = 0;
//    node11->childs = NULL;
//    EFGNODE *dev_node11 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node11, efg_node_size));
//    // node 12
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node12 = (EFGNODE *) malloc(efg_node_size);
//    if (!node12) {
//        printf("Error malloc node12\n"); return 1;
//    }
//    node12->parent = dev_node4;
//    node12->player = 0;
//    node12->value = -1.0;
//    node12->information_set = NULL;
//    node12->childs_count = 0;
//    node12->childs = NULL;
//    EFGNODE *dev_node12 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node12, efg_node_size));
//    // node 13
//    efg_node_size = sizeof(EFGNODE);
//    EFGNODE *node13 = (EFGNODE *) malloc(efg_node_size);
//    if (!node13) {
//        printf("Error malloc node13\n"); return 1;
//    }
//    node13->parent = dev_node4;
//    node13->player = 0;
//    node13->value = 0.0;
//    node13->information_set = NULL;
//    node13->childs_count = 0;
//    node13->childs = NULL;
//    EFGNODE *dev_node13 = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node13, efg_node_size));
//
//    // Childs
//    // node1
//    size_t efgnode_childs_size = 3 * sizeof(EFGNODE**);
//    EFGNODE **node1_childs = (EFGNODE**) malloc(efgnode_childs_size);
//    node1_childs[0] = dev_node2;
//    node1_childs[1] = dev_node3;
//    node1_childs[2] = dev_node4;
//    EFGNODE **dev_node1_childs = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node1_childs, efgnode_childs_size));
//    CHECK_ERROR(cudaMemcpy(dev_node1_childs, node1_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
//    node1->childs = dev_node1_childs;
//    // TODO update childs_cnt here
//    // node2
//    efgnode_childs_size = 3 * sizeof(EFGNODE**);
//    EFGNODE **node2_childs = (EFGNODE**) malloc(efgnode_childs_size);
//    node2_childs[0] = dev_node5;
//    node2_childs[1] = dev_node6;
//    node2_childs[2] = dev_node7;
//    EFGNODE **dev_node2_childs = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node2_childs, efgnode_childs_size));
//    CHECK_ERROR(cudaMemcpy(dev_node2_childs, node2_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
//    node2->childs = dev_node2_childs;
//    // node3
//    efgnode_childs_size = 3 * sizeof(EFGNODE**);
//    EFGNODE **node3_childs = (EFGNODE**) malloc(efgnode_childs_size);
//    node3_childs[0] = dev_node8;
//    node3_childs[1] = dev_node9;
//    node3_childs[2] = dev_node10;
//    EFGNODE **dev_node3_childs = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node3_childs, efgnode_childs_size));
//    CHECK_ERROR(cudaMemcpy(dev_node3_childs, node3_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
//    node3->childs = dev_node3_childs;
//    // node4
//    efgnode_childs_size = 3 * sizeof(EFGNODE**);
//    EFGNODE **node4_childs = (EFGNODE**) malloc(efgnode_childs_size);
//    node4_childs[0] = dev_node11;
//    node4_childs[1] = dev_node12;
//    node4_childs[2] = dev_node13;
//    EFGNODE **dev_node4_childs = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_node4_childs, efgnode_childs_size));
//    CHECK_ERROR(cudaMemcpy(dev_node4_childs, node4_childs, efgnode_childs_size, cudaMemcpyHostToDevice));
//    node4->childs = dev_node4_childs;
//
//    // copy nodes to GPU memory
//    CHECK_ERROR(cudaMemcpy(dev_node1, node1, efg_node_size, cudaMemcpyHostToDevice));
//
//    CHECK_ERROR(cudaMemcpy(dev_node2, node2, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node3, node3, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node4, node4, efg_node_size, cudaMemcpyHostToDevice));
//
//    CHECK_ERROR(cudaMemcpy(dev_node5, node5, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node6, node6, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node7, node7, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node8, node8, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node9, node9, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node10, node10, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node11, node11, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node12, node12, efg_node_size, cudaMemcpyHostToDevice));
//    CHECK_ERROR(cudaMemcpy(dev_node13, node13, efg_node_size, cudaMemcpyHostToDevice));
//
//    // Prepare a list of terminal nodes
//    int terminal_nodes_size = 9;
//    size_t terminal_nodes_ptr_size = sizeof(EFGNODE**) * terminal_nodes_size;
//    EFGNODE **terminal_nodes = (EFGNODE**) malloc(terminal_nodes_ptr_size);
//    if (!terminal_nodes) {
//        printf("error malloc terminal nodes\n");
//        return 1;
//    }
//    terminal_nodes[0] = dev_node5;
//    terminal_nodes[1] = dev_node6;
//    terminal_nodes[2] = dev_node7;
//    terminal_nodes[3] = dev_node8;
//    terminal_nodes[4] = dev_node9;
//    terminal_nodes[5] = dev_node10;
//    terminal_nodes[6] = dev_node11;
//    terminal_nodes[7] = dev_node12;
//    terminal_nodes[8] = dev_node13;
//    EFGNODE** dev_terminal_nodes = (EFGNODE**) malloc(terminal_nodes_ptr_size);
//    CHECK_ERROR(cudaMalloc((void **) &dev_terminal_nodes, terminal_nodes_ptr_size));
//    CHECK_ERROR(cudaMemcpy(dev_terminal_nodes, terminal_nodes, terminal_nodes_ptr_size, cudaMemcpyHostToDevice));
//
//    // information sets lengths by depth
//    size_t information_sets_lengths_size = 2*sizeof(int);
//    int *information_sets_lengths = (int *) malloc(information_sets_lengths_size);
//    if (!information_sets_lengths) {
//        printf("Error malloc information_sets_lengths\n");
//        return 1;
//    }
//    information_sets_lengths[0] = 1;
//    information_sets_lengths[1] = 1;
//    int *dev_information_sets_lengths_size = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_information_sets_lengths_size, information_sets_lengths_size));
//    CHECK_ERROR(cudaMemcpy(dev_information_sets_lengths_size, information_sets_lengths, information_sets_lengths_size, cudaMemcpyHostToDevice));
//    // EFG nodes by lengths
//    size_t efg_nodes_lengths_size = 2*sizeof(int);
//    int *efg_nodes_lengths = (int *) malloc(efg_nodes_lengths_size);
//    if (!efg_nodes_lengths) {
//        printf("Error malloc efg_nodes_lengths\n");
//        return 1;
//    }
//    efg_nodes_lengths[0] = 1;
//    efg_nodes_lengths[1] = 3;
//    int *dev_efg_nodes_lengths_size = NULL;
//    CHECK_ERROR(cudaMalloc((void **) &dev_efg_nodes_lengths_size, efg_nodes_lengths_size));
//    CHECK_ERROR(cudaMemcpy(dev_efg_nodes_lengths_size, efg_nodes_lengths, efg_nodes_lengths_size, cudaMemcpyHostToDevice));
//
//    // kernel - counterfactual value computation
//    cfv_kernel<<<1, 32>>>(dev_terminal_nodes, terminal_nodes_size);
//
//    /* FREE MEMORY */
//    free(information_set1);
//    free(information_set2);
//
//    free(node1);
//
//    free(node2);
//    free(node3);
//    free(node4);
//
//    free(node5);
//    free(node6);
//    free(node7);
//    free(node8);
//    free(node9);
//    free(node10);
//    free(node11);
//    free(node12);
//    free(node13);
//
//    free(node1_childs);
//
//    free(node2_childs);
//    free(node3_childs);
//    free(node4_childs);
//
//    free(terminal_nodes);
//
//    CHECK_ERROR(cudaFree(dev_information_set1));
//    CHECK_ERROR(cudaFree(dev_information_set2));
//
//    CHECK_ERROR(cudaFree(dev_node1));
//
//    CHECK_ERROR(cudaFree(dev_node2));
//    CHECK_ERROR(cudaFree(dev_node3));
//    CHECK_ERROR(cudaFree(dev_node4));
//
//    CHECK_ERROR(cudaFree(dev_node5));
//    CHECK_ERROR(cudaFree(dev_node6));
//    CHECK_ERROR(cudaFree(dev_node7));
//    CHECK_ERROR(cudaFree(dev_node8));
//    CHECK_ERROR(cudaFree(dev_node9));
//    CHECK_ERROR(cudaFree(dev_node10));
//    CHECK_ERROR(cudaFree(dev_node11));
//    CHECK_ERROR(cudaFree(dev_node12));
//    CHECK_ERROR(cudaFree(dev_node13));
//
//    CHECK_ERROR(cudaFree(dev_node1_childs));
//    CHECK_ERROR(cudaFree(dev_node2_childs));
//    CHECK_ERROR(cudaFree(dev_node3_childs));
//
//    CHECK_ERROR(cudaFree(dev_terminal_nodes));

    return 0;
}