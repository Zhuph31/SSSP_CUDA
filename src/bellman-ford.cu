#include "common.cuh"
#include "bellman-ford.cuh"

#include <stdio.h>

__global__ void bf_iter(CSRGraph g, edge_data_type* last_cost, edge_data_type* new_cost) {
    index_type source_vtx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (source_vtx < g.nnodes) {
        // Only process nodes that changed last time, otherwise it runs forever
        if (last_cost[source_vtx] == new_cost[source_vtx]) {
            return;
        }
        atomicMin(&new_cost[source_vtx], last_cost[source_vtx]);

        for (int e = g.row_start[source_vtx]; e < g.row_start[source_vtx + 1]; e++) {
            edge_data_type source_cost = last_cost[source_vtx];
            edge_data_type edge_weight = g.edge_data[e];
            index_type dest_vtx = g.edge_dst[e];
            edge_data_type old_dest_cost = last_cost[dest_vtx];
            edge_data_type new_dest_cost = edge_weight + source_cost;
            // Check if the distance is already set to max then just take the max since,
            if (source_cost >= MAX_VAL){
                new_dest_cost = MAX_VAL;
            }

            if (new_dest_cost < old_dest_cost) {
                atomicMin(&new_cost[dest_vtx],new_dest_cost);
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
