//----------------------------------------------------------------------------------------
/*!
 * \file       information_set.h
 * \author     Jan Rudolf
 * \date       2021/01/14
 * \brief      Information set from EFG formalism.
 *
 *  Class implementing an information set from EFG formalism. The file contains INFORMATION_SET (dev) type and  InformationSet (host) class.
 *
*/
//----------------------------------------------------------------------------------------
#ifndef INFORMATION_SET_H_
#define INFORMATION_SET_H_
#include <iostream>
#include <vector>

#include "cuda_base.h"

//! GPU implementation of information set is an array of floats.
typedef float INFORMATION_SET;

//! This class represents an information set from the EFG formalism.
/*!
 This class represents an information set from the EFG formalism.
 * */
class InformationSet {
private:
    size_t hash_t_ = 0;
    unsigned int number_of_actions_ = 0;

    size_t information_set_t_size_ = 0;
    INFORMATION_SET* information_set_t_ = NULL;

    INFORMATION_SET* dev_information_set_t_ = NULL;

    void init_();

public:
    //! Constructor for an information set.
    /*!
     * Constructor for an information set.
     * @param hash Unique hash of an information set exported from GTLib.
     * @param number_of_actions Number of actions of nodes in the information set.
     */
    InformationSet(size_t hash, unsigned number_of_actions);
    //! Copies the information set from the host to the GPU.
    /*!
     *  Copies the information set's data from the host memory to the GPU memory.
     */
    void memcpy_host_to_gpu ();
    //! Copies the information set from the GPU to the host memory.
    /*!
     *  Copies the information set's data from GPU global memory to the host memory.
     */
    void memcpy_gpu_to_host ();

    //! Returns a pointer of INFORMATION_SET array on GPU. INFORMATION_SET is just renamed float.
    /*!
     * Returns a pointer of INFORMATION_SET array on GPU. INFORMATION_SET is just renamed float.
     */
    INFORMATION_SET* get_gpu_ptr();
    //! Returns the unique hash of the information set.
    /*!
     * Returns the unique hash of the information set.
     */
    size_t get_hash();
    //! Returns the current strategy of the information set.
    /*!
     * Returns a vector of current strategy stored in the information set. The size of the vector is the same as number of actions.
     */
    std::vector<double> get_current_strategy();
    //! Returns the regrets of the information set.
    /*!
     * Returns a vector of regrets stored in the information set. The size of the vector is the same as number of actions.
     */
    std::vector<double> get_regrets();
    //! Returns the counterfactual values of the information set.
    /*!
     * Returns a vector of counterfactual values stored in the information set. The size of the vector is the same as number of actions.
     */
    std::vector<double> get_cfv();
    //! Returns the reach probability of the information set.
    /*!
     * //! Returns the reach probability of the information set.
     */
    float get_reach_probability ();
    //! Returns the average strategy of the information set.
    /*!
     * Returns a vector of average strategy stored in the information set. The size of the vector is the same as number of actions.
     */
    std::vector<double> get_average_strategy();

    ~InformationSet();
};

#endif // INFORMATION_SET_H_