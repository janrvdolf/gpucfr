#include <iostream>

#include "gpucfr.h"


int main () {
    /* Goofspiel 2 */
    unsigned int iterations = 1000;
    GPUCFR instance1 = GPUCFR("../data/gs2.game");
    instance1.run_iterations(iterations);
    std::cout << "1000 iterations takes " << instance1.elapsed_time() << "ms on Goospiel 2" << std::endl;
return 0;
    iterations = 5000;
    GPUCFR instance2 = GPUCFR("../data/gs2.game");
    instance2.run_iterations(iterations);
    std::cout << "5000 iterations takes " << instance2.elapsed_time() << "ms on Goospiel 2" << std::endl;

    iterations = 10000;
    GPUCFR instance3 = GPUCFR("../data/gs2.game");
    instance3.run_iterations(iterations);
    std::cout << "10000 iterations takes " << instance3.elapsed_time() << "ms on Goospiel 2" << std::endl;

    /* Goofspiel 3 */
    iterations = 1000;
    GPUCFR instance4 = GPUCFR("../data/gs3.game");
    instance4.run_iterations(iterations);
    std::cout << "1000 iterations takes " << instance4.elapsed_time() << "ms on Goospiel 3" << std::endl;

    iterations = 5000;
    GPUCFR instance5 = GPUCFR("../data/gs3.game");
    instance5.run_iterations(iterations);
    std::cout << "5000 iterations takes " << instance5.elapsed_time() << "ms on Goospiel 3" << std::endl;

    iterations = 10000;
    GPUCFR instance6 = GPUCFR("../data/gs3.game");
    instance6.run_iterations(iterations);
    std::cout << "10000 iterations takes " << instance6.elapsed_time() << "ms on Goospiel 3" << std::endl;

    /* Goofspiel 3 */
    iterations = 1000;
    GPUCFR instance7 = GPUCFR("../data/gs4.game");
    instance7.run_iterations(iterations);
    std::cout << "1000 iterations takes " << instance7.elapsed_time() << "ms on Goospiel 4" << std::endl;

    iterations = 5000;
    GPUCFR instance8 = GPUCFR("../data/gs4.game");
    instance8.run_iterations(iterations);
    std::cout << "5000 iterations takes " << instance8.elapsed_time() << "ms on Goospiel 4" << std::endl;

    iterations = 10000;
    GPUCFR instance9 = GPUCFR("../data/gs4.game");
    instance9.run_iterations(iterations);
    std::cout << "10000 iterations takes " << instance9.elapsed_time() << "ms on Goospiel 4" << std::endl;

    return 0;
}