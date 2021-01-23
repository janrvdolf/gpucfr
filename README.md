The project implements Counterfactual Regret Minimization in parallel on a CUDA-compatible GPU.

The class GPUCFR implements the GPU version of CFR. Classes Node and InformationSet are support classes for EFG formalism. Struct efg_node_t represents the Node class on the GPU. 

Header files are in the directory include/, and source code files are in the directory src/. Data directory contains exported EFG trees for three variants of Goofspiel.

The code is tested on Ubuntu 20.04 with CUDA 11.

**WARNING** The code does not use Chance for now. 