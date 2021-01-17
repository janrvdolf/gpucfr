#include <iostream>

#include "gpucfr.h"


int main (int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "Run the program with two parameters - <string: efg game file path>" << std::endl;
        return 1;
    }
    std::string gamepath(argv[1]);
    /* Goofspiel 2 */
    std::cout << "Outputs average strategy on Goofspiel 2:" << std::endl;

    unsigned int iterations = 1000;
    GPUCFR instance = GPUCFR(gamepath);
    instance.run_iterations(iterations);

    instance.print_average_strategy();

    return 0;
}