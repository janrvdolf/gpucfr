#ifndef INFORMATION_SET_H_
#define INFORMATION_SET_H_
#include <iostream>
#include <vector>

#include "cuda_base.h"


typedef float INFORMATION_SET;

class InformationSet {
private:
    size_t hash_t_ = 0;
    unsigned int number_of_actions_ = 0;

    size_t information_set_t_size_ = 0;
    INFORMATION_SET* information_set_t_ = NULL;

    INFORMATION_SET* dev_information_set_t_ = NULL;

    void init_();

public:
    InformationSet(size_t hash, unsigned number_of_actions);

    void memcpy_host_to_gpu ();

    void memcpy_gpu_to_host ();

    INFORMATION_SET* get_gpu_ptr();

    size_t get_hash();

    std::vector<double> get_current_strategy();

    std::vector<double> get_regrets();

    std::vector<double> get_cfv();

    float get_reach_probability ();

    std::vector<double> get_average_strategy();

    ~InformationSet();
};

#endif // INFORMATION_SET_H_