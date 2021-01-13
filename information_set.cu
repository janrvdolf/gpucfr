#include "information_set.h"

void InformationSet::init_(){
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

InformationSet::InformationSet(size_t hash, unsigned number_of_actions) {
        hash_t_ = hash;
        number_of_actions_ = number_of_actions;

        information_set_t_size_ = (4 * number_of_actions_ + 2) * sizeof(INFORMATION_SET);
        information_set_t_ = (INFORMATION_SET*) malloc(information_set_t_size_);

        init_();

        CHECK_ERROR(cudaMalloc((void **) &dev_information_set_t_, information_set_t_size_));
    }

void InformationSet::memcpy_host_to_gpu () {
        // copy data from CPU's RAM to GPU's global memory
        CHECK_ERROR(cudaMemcpy(dev_information_set_t_, information_set_t_, information_set_t_size_, cudaMemcpyHostToDevice));
    }

void InformationSet::memcpy_gpu_to_host () {
        // copy data from GPU's global memory to CPU's RAM
        CHECK_ERROR(cudaMemcpy(information_set_t_, dev_information_set_t_, information_set_t_size_, cudaMemcpyDeviceToHost));
    }

    INFORMATION_SET* InformationSet::get_gpu_ptr() {
        return dev_information_set_t_;
    }

    size_t InformationSet::get_hash() {
        return hash_t_;
    }

    std::vector<double> InformationSet::get_current_strategy() {
        std::vector<double> returning_strategy;
        int offset = 2;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    std::vector<double> InformationSet::get_regrets() {
        std::vector<double> returning_strategy;
        int offset = 2+3*number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    std::vector<double> InformationSet::get_cfv() {
        std::vector<double> returning_strategy;
        int offset = 2+2*number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

    float InformationSet::get_reach_probability () {
        return information_set_t_[1];
    }

    std::vector<double> InformationSet::get_average_strategy() {
        std::vector<double> returning_strategy;
        unsigned int offset = 2 + number_of_actions_;
        for (unsigned int i = offset; i < offset + number_of_actions_; i++) {
            returning_strategy.push_back(information_set_t_[i]);
        }
        return returning_strategy;
    }

InformationSet::~InformationSet() {
        free(information_set_t_);
        information_set_t_ = NULL;

        CHECK_ERROR(cudaFree(dev_information_set_t_));
        dev_information_set_t_ = NULL;
    }