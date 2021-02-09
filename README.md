The GPUCFR project implements Counterfactual Regret Minimization (CFR) [1] in parallel on a CUDA-compatible NVIDIA GPU. GPUCFR uses simultaneous updates Vanilla CFR on extensive-form games. The project started as a semestral project during the General-purpose computing on GPUs course at Czech Technical University in Prague.

The class *GPUCFR* implements the GPU version of CFR. Classes *Node* and *InformationSet* are support classes for EFG formalism. Struct *efg_node_t* represents the *Node* class on the GPU. 

Header files are in the directory *include*, and source code files are in the directory *src*. Data directory contains exported EFG trees for three variants of Goofspiel.

The code is tested on Ubuntu 20.04 with CUDA 11.

**TODO** 
- add a Chance node
- more thoroughly check inputs and errors
- fail fast if the host does not have CUDA GPU

References:

[1] Martin Zinkevich et al. “Regret Minimization in Games with Incomplete Information”. In: Advances in Neural Information Processing Systems 20. Ed. by J. C. Platt et al. Curran Associates, Inc., 2008, pp. 1729–1736. url: http://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf
