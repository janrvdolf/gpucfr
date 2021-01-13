#include <iostream>

#include "gpucfr.h"


int main () {
    /* Goofspiel 2 */
    unsigned int iterations = 1000;
    GPUCFR instance1 = GPUCFR("../gs2.game");
    instance1.run_iterations(iterations);
    instance1.export_strategy_for_gtlib("gs2.strategy");
    std::cout << "1000 iterations takes " << instance1.elapsed_time() << "ms on Goospiel 2" << std::endl;

//    iterations = 5000;
//    GPUCFR game_loader2 = GPUCFR("../gs2.game");
//    game_loader2.load();
//    game_loader2.memcpy_host_to_gpu();
//    game_loader2.run_iterations(iterations);
//    game_loader2.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader2.elapsed_time() << "ms on Goospiel 2" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader3 = GPUCFR("../gs2.game");
//    game_loader3.load();
//    game_loader3.memcpy_host_to_gpu();
//    game_loader3.run_iterations(iterations);
//    game_loader3.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader3.elapsed_time() << "ms on Goospiel 2" << std::endl;
//
//    /* Goofspiel 3 */
//    iterations = 1000;
//    GPUCFR game_loader4 = GPUCFR("../gs3.game");
//    game_loader4.load();
//    game_loader4.memcpy_host_to_gpu();
//    game_loader4.run_iterations(iterations);
//    game_loader4.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader4.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    iterations = 5000;
//    GPUCFR game_loader5 = GPUCFR("../gs3.game");
//    game_loader5.load();
//    game_loader5.memcpy_host_to_gpu();
//    game_loader5.run_iterations(iterations);
//    game_loader5.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader5.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader6 = GPUCFR("../gs3.game");
//    game_loader6.load();
//    game_loader6.memcpy_host_to_gpu();
//    game_loader6.run_iterations(iterations);
//    game_loader6.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader6.elapsed_time() << "ms on Goospiel 3" << std::endl;
//
//    /* Goofspiel 3 */
//    iterations = 1000;
//    GPUCFR game_loader7 = GPUCFR("../gs4.game");
//    game_loader7.load();
//    game_loader7.memcpy_host_to_gpu();
//    game_loader7.run_iterations(iterations);
//    game_loader7.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader7.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
//    iterations = 5000;
//    GPUCFR game_loader8 = GPUCFR("../gs4.game");
//    game_loader8.load();
//    game_loader8.memcpy_host_to_gpu();
//    game_loader8.run_iterations(iterations);
//    game_loader8.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader8.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
//    iterations = 10000;
//    GPUCFR game_loader9 = GPUCFR("../gs4.game");
//    game_loader9.load();
//    game_loader9.memcpy_host_to_gpu();
//    game_loader9.run_iterations(iterations);
//    game_loader9.memcpy_gpu_to_host();
//    std::cout << "1000 iterations takes " << game_loader9.elapsed_time() << "ms on Goospiel 4" << std::endl;
//
    return 0;
}