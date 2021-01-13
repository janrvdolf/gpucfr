#include "efg_node.h"


Node::Node(Node *parent, InformationSet *information_set) {
        parent_ = parent;
        information_set_ = information_set;

        size_t node_t_size = sizeof(EFGNODE);
        node_t_ = (EFGNODE*) malloc(node_t_size);

        CHECK_ERROR(cudaMalloc((void **) &dev_node_t_, node_t_size));
    }

Node::~Node() {
        if (node_t_)
            free(node_t_);
        if (dev_node_t_)
            CHECK_ERROR(cudaFree(dev_node_t_));
        if (dev_children_)
            CHECK_ERROR(cudaFree(dev_children_));
    }

    EFGNODE* Node::get_gpu_ptr() {
        return dev_node_t_;
    }

    void Node::memcpy_host_to_gpu () {
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

    void Node::update_gtlib_data(
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

    void Node::add_child(Node *child) {
        children.push_back(child);
    }
