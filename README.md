The GPUCFR project implements Counterfactual Regret Minimization (CFR) [1] in parallel on a CUDA-compatible NVIDIA GPU. GPUCFR uses simultaneous updates Vanilla CFR on extensive-form games. The project started as a semestral project during the General-purpose computing on GPUs [2] course at Czech Technical University in Prague.

The class *GPUCFR* implements the GPU version of CFR. Classes *Node* and *InformationSet* are support classes for EFG formalism. Struct *efg_node_t* represents the *Node* class on the GPU. 

Header files are in the directory *include*, and source code files are in the directory *src*. Data directory contains exported EFG trees for three variants of Goofspiel.

The code is tested on a desktop computer with Ubuntu 20.04, CUDA 11, and GeForce GTX 1050 Mobile. Also, the code was run on a cluster with CUDA 11 and Tesla V100. 

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

[1] Martin Zinkevich et al. “Regret Minimization in Games with Incomplete Information”. In: Advances in Neural Information Processing Systems 20. Ed. by J. C. Platt et al. Curran Associates, Inc., 2008, pp. 1729–1736. url: http://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf

[2] Jan Rudolf "Counterfactual Regret Minimization on GPU". In: General-purpose computing on GPUs, Faculty of Electrial Engineering, Czech Technical University in Prague. url: https://cent.felk.cvut.cz/courses/GPU/archives/2020-2021/W/rudolja1/
