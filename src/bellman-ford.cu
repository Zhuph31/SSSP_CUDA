#include "common.cuh"
#include "bellman-ford.cuh"

#include <stdio.h>

__global__ void bf_iter(CSRGraph g, edge_data_type* last_d, edge_data_type* new_d) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < g.nnodes) {
        // Only process nodes that changed last time
        if (last_d[index] == new_d[index]) {
            return;
        }
        atomicMin(&new_d[index], last_d[index]);

        // somewhat baed on https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b
        for (int j = g.row_start[index]; j < g.row_start[index + 1]; j++) {
            edge_data_type w = last_d[index];
            edge_data_type ew = g.edge_data[j];
            index_type n = g.edge_dst[j];
            edge_data_type nw = last_d[n];
            edge_data_type new_w = ew + w;
            // Check if the distance is already set to max then just take the max since,
            if (w >= MAX_VAL){
                new_w = MAX_VAL;
            }

            if (new_w < nw) {
                atomicMin(&new_d[n],new_w);
            }
        }
    }
}


void bellman_ford(CSRGraph& g, edge_data_type* dists) {
    double start,end = 0;
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d, *d_d1, *d_d2  = NULL;
    check_cuda(cudaMalloc(&d_d1, g.nnodes * sizeof(edge_data_type)));
    check_cuda(cudaMalloc(&d_d2, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node = 0. Otherwise need to change this
    check_cuda(cudaMemset(&d_d1[1], 0xFF,  (g.nnodes-1) * sizeof(edge_data_type)));
    check_cuda(cudaMemset(&d_d2[0], 0xFF,  (g.nnodes) * sizeof(edge_data_type)));

    start = getTimeStamp();
    for (int i = 0; i < g.nnodes -1; i++) {
        // printf("Iter %d\n",i);
        edge_data_type* tmp = d_d1;
        d_d1 = d_d2;
        d_d2 = tmp;
 
        bf_iter<<<(g.nnodes + 512 - 1) / 512,512>>>(d_g, d_d1, d_d2);
    }
    d_d = d_d2;
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}
