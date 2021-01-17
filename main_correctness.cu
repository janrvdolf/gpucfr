#include <iostream>

#include "gpucfr.h"


int main () {
    /* Goofspiel 2 */
    std::cout << "Outputs average strategy on Goofspiel 2:" << std::endl;

    unsigned int iterations = 1000;
    GPUCFR instance = GPUCFR("../data/gs2.game");
    instance.run_iterations(iterations);

    instance.print_average_strategy();

    return 0;
}