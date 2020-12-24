#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Simple function to check for CUDA runtime errors.
static void handleCUDAError(
        cudaError_t error,		// error code
        const char *file,		// file within error was generated
        int line )			// line where error occurs
{
    if (error != cudaSuccess) {	// any error -> display error message and terminate application
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define CHECK_ERROR( error ) ( handleCUDAError( error, __FILE__, __LINE__ ) )

int main() {
    std::cout << "Hello, World!" << std::endl;

    int *dev_in_data = NULL;

    CHECK_ERROR(cudaMalloc((void **) &dev_in_data, sizeof(int)));

    /*
     Postavim strom na GPU u tak ze udelam tolik nodu, kolik je ve strome. Jeden node bude structura, ve ktere bude ukazatel na parent strukturu.
     Tyto struktury musi byt alokovany na gpu. Udelam spojeni EFG node -> Structura uzlu. Zavolam threads na kazdy uzel, while smycka zkonci v root stromu.
     Postavim ho tak, ze jak budu prochazet/stavet EFG strom tak se bude stavet i strom na GPU.

     Pocet infosetu v ramci uzlu uz mam zjisteni z diplomky.

     Vystavim pole hodnot pro current strategie atd. viz tabule/fotka v mobilu.

     Po zkonceni se zkopiruji je strategie. Jak to spustit jako 2D mrizku? Udelat is experiment - > vygeneruju 2D pole struktur s hodnotami column a sloupec
    a porovnam.
     * */

    return 0;
}
