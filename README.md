The GPUCFR project implements Counterfactual Regret Minimization (CFR) [1] in parallel on a CUDA-compatible NVIDIA GPU. GPUCFR uses simultaneous updates Vanilla CFR on extensive-form games. The project started as a semestral project during the General-purpose computing on GPUs [2] course at Czech Technical University in Prague. Read report https://github.com/janrvdolf/gpucfr/blob/master/report.pdf

The class *GPUCFR* implements the GPU version of CFR. Classes *Node* and *InformationSet* are support classes for EFG formalism. Struct *efg_node_t* represents the *Node* class on the GPU. 

Header files are in the directory *include*, and source code files are in the directory *src*. Data directory contains exported EFG trees for three variants of Goofspiel.

The code is tested on a desktop computer with Ubuntu 20.04, CUDA 11, and NVIDIA GeForce GTX 1050 Mobile. Also, the code was run on a cluster with NVIDIA Tesla V100. 

Compilation on CTU's Research Center for Informatics Cluster
============================================================
Ask for an interactive computation node with GPU:

    srun -p gpufast --gres=gpu:1 --pty bash -i

Add dependencies:

    module add CMake
    module add fosscuda/2020a

Create a compilation directory:

    mkdir cmake-build-debug && cd cmake-build-debug

Compile (with the choice of Volta architecture for NVIDIA Tesla V100):

    cmake -DCMAKE_CUDA_ARCHITECTURES="70" ..
    make

Compilation on Cesnet MetaCentrum
=================================
Ask for an interactive computation node with GPU:

    qsub -I -l select=1:ncpus=1:ngpus=1 -q gpu

Add dependencies:

    module add cmake
    module add cuda

Create a compilation directory:

    mkdir cmake-build-debug && cd cmake-build-debug

Compile:

    cmake ..
    make

References
==========

[1] Martin Zinkevich et al. “Regret Minimization in Games with Incomplete Information”. Advances in Neural Information Processing Systems 20. Ed. by J. C. Platt et al. Curran Associates, Inc., 2008, pp. 1729–1736. URL: http://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf

[2] João Reis "A GPU implementation of Counterfactual Regret Minimization". Master thesis, University of Porto, 2015. URL: https://repositorio-aberto.up.pt/handle/10216/83517

[3] Jan Rudolf "Counterfactual Regret Minimization on GPU". General-purpose programming on GPUs, Archive 2020-2021, Faculty of Electrical Engineering, Czech Technical University in Prague, 2021. URL: https://cent.felk.cvut.cz/courses/GPU/archives/2020-2021/W/rudolja1/

Cited by
========
Juho Kim "GPU-Accelerated Counterfactual Regret Minimization". 2024. URL: https://arxiv.org/abs/2408.14778
