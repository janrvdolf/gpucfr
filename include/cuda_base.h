//----------------------------------------------------------------------------------------
/*!
 * \file       cuda_base.h
 * \author     Jan Rudolf
 * \date       2021/01/14
 * \brief      Import CUDA related header files and useful utils.
 *
 *  Import CUDA related header files and useful utils.
 *
*/
//----------------------------------------------------------------------------------------
#ifndef CUDA_BASE_H_
#define CUDA_BASE_H_
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/* SOURCE
 * The following code adopted from Ing. Jaroslav Sloup.
 * https://cent.felk.cvut.cz/courses/GPU/seminars.html
*/

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

//! Macro for catching CUDA error.
#define CHECK_ERROR( error ) ( handleCUDAError( error, __FILE__, __LINE__ ) )

#endif // CUDA_BASE_H_