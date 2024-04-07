#ifndef COMMON_H
#define COMMON_H

#include <sys/time.h>
#include <stdio.h>
#include <fcntl.h>

//TODO: make this template data
typedef unsigned index_type; // should be size_t, but GPU chokes on size_t
typedef unsigned edge_data_type;
typedef int node_data_type;
const edge_data_type MAX_VAL = UINT_MAX;




inline double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

static void check_cuda_error(const cudaError_t e, const char *file, const int line)
{
  if (e != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s (%d)\n", file, line, cudaGetErrorString(e), e);
    exit(1);
  }
}
template <typename T>
static void check_retval(const T retval, const T expected, const char *file, const int line) {
  if(retval != expected) {
    fprintf(stderr, "%s:%d: Got %d, expected %d\n", file, line, retval, expected);
    exit(1);
  }
}


inline static __device__ __host__ int roundup(int a, int r) {
  return ((a + r - 1) / r) * r;
}

inline static __device__ __host__ int GG_MIN(int x, int y) {
  if(x > y) return y; else return x;
}

#define check_cuda(x) check_cuda_error(x, __FILE__, __LINE__)
#define check_rv(r, x) check_retval(r, x, __FILE__, __LINE__)



#if CUDART_VERSION >= 4000
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
#else
#define CUDA_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
#endif


static inline bool compare(edge_data_type* cpu, edge_data_type* gpu, index_type n) {

    for (int i = 0; i < n; i++) {
        if (cpu[i] != gpu[i]) {
            printf("Wrong at %d: %d!=%d\n",i,cpu[i],gpu[i]);
            return false;
        }
    }

    printf("Match!\n");
    return true;
}


#endif