#ifndef COMMON_H
#define COMMON_H

#include <sys/time.h>

//TODO: make this template data
typedef unsigned index_type; // should be size_t, but GPU chokes on size_t
typedef unsigned edge_data_type;
typedef int node_data_type;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

#if CUDART_VERSION >= 4000
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
#else
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
#endif

#  define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = CUDA_DEVICE_SYNCHRONIZE();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#endif