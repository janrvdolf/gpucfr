#ifndef EFG_NODE_H_
#define EFG_NODE_H_
#include <vector>

#include "cuda_base.h"
#include "information_set.h"

//! The struct represents the GPU implementation of Node class.
typedef struct efg_node_t {
    struct efg_node_t *parent;             ///< A pointer to the parent node.

    int player;                            ///< Number of player.
    float reach_probability;               ///< Reach probability of the node.
    float value;                           ///< Values/Counterfactual value of the node.
    INFORMATION_SET *information_set;      ///< A pointer to the information set.

    // children
    int childs_count;                      ///< Number of children.
    struct efg_node_t **childs;            ///< Array of children.
} EFGNODE;


//! This class represents an EFG node from the EFG formalism.
/*!
 This class represents an EFG node from the EFG formalism.
 * */
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

    std::vector<Node*> children;
public:

    //! Constructor for an EFG node.
    /*!
     * Construction for an EFG node represents a building block for a game tree.
     * @param parent A link for a parent EFG node.
     * @param information_set A link to the information set the EFG node is in.
     */
    Node(Node *parent, InformationSet *information_set);

    ~Node();

    //! Returns a pointer of EFGNODE on GPU.
    /*!
     * Return a pointer of the struct EFGNODE allocated on GPU corresponding to the class instance.
     */
    EFGNODE* get_gpu_ptr();

    //! Copies the node from the host to the GPU.
    /*!
     * Allocates memory and copies the data to the GPU. On GPU, the class is represented by EFGNODE.
     */
    void memcpy_host_to_gpu ();

    //! Assign data exported from GTLib that corresponds to this EFG NODE.
    /*!
     *
     * @param hash Unique hash of the node from GTLib.
     * @param number_of_actions Number of actions that are available in this node.
     * @param player Which player is the decision maker in this node.
     * @param parent_hash Unique hash of the parent node from GTLib.
     * @param information_set_hash Unique hash of information set where the node belongs.
     * @param value If the node is a terminal node, values is an utility for player 1. Zero otherwise.
     */
    void update_gtlib_data(
            size_t   hash,
            unsigned int    number_of_actions,
            unsigned int    player,
            size_t   parent_hash,
            size_t   information_set_hash,
            float           value);

    //! Adds a child to the children of the node.
    /*!
     *
     * @param child A pointer to the child node.
     */
    void add_child(Node *child);
};

#endif // EFG_NODE_H_