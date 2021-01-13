#ifndef EFG_NODE_H_
#define EFG_NODE_H_
#include <vector>

#include "cuda_base.h"
#include "information_set.h"


typedef struct efg_node_t {
    struct efg_node_t *parent;

    int player;
    float reach_probability;
    float value;
    INFORMATION_SET *information_set;

    // children
    int childs_count;
    struct efg_node_t **childs;
} EFGNODE;

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

    Node(Node *parent, InformationSet *information_set);

    ~Node();

    EFGNODE* get_gpu_ptr();

    void memcpy_host_to_gpu ();

    void update_gtlib_data(
            size_t   hash,
            unsigned int    number_of_actions,
            unsigned int    player,
            size_t   parent_hash,
            size_t   information_set_hash,
            float           value);

    void add_child(Node *child);
};

#endif // EFG_NODE_H_