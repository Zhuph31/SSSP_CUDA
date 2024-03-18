#include "csr.h"
#include <sys/time.h>


double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}


__global__ const edge_data_type MAX_VAL = UINT_MAX;
__global__ void nf_iter(CSRGraph g, edge_data_type* d, edge_data_type delta, index_type* last_near_pile, index_type last_near_len, index_type* new_near_pile, 
index_type* new_near_len, index_type* new_far_pile, index_type* new_far_len) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    //only work on near pile
    if (index < last_near_len) {
        index_type s_idx = last_near_pile[index];
        // somewhat baed on https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b
        for (int j = g.row_start[s_idx]; j < g.row_start[s_idx + 1]; j++) {
            edge_data_type w = d[s_idx];
            edge_data_type ew = g.edge_data[j];
            index_type n = g.edge_dst[j];
            edge_data_type nw = d[n];
            edge_data_type new_w = ew + w;
            // Check if the distance is already set to max then just take the max since,
            if (w >= MAX_VAL){
                new_w = MAX_VAL;
            }

            if (new_w < nw) {
                atomicMin(&d[n],new_w);
                //seperating into near and far pile 
                index_type q_idx;
                if (new_w < delta){
                    q_idx = atomicAdd(new_near_len,1);
                    new_near_pile[q_idx] = n;
                }
                else{
                    q_idx = atomicAdd(new_far_len,1);
                    new_far_pile[q_idx] = n;
                }
                // printf("Adding %d to q[%d]\n",n,q_idx);
            }
        }
    }
}


__global__ void far_split(edge_data_type* d, edge_data_type delta, index_type* last_far_pile, index_type last_far_len, index_type* new_near_pile, 
index_type* new_near_len, index_type* new_far_pile, index_type* new_far_len) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
    //only split far pile into near and far
    if (index < last_far_len) {
        index_type far_idx = last_far_pile[index];
        edge_data_type nw = d[far_idx];
        //printf("index %d, far_idx %d, distance %d\n", index, far_idx, nw);
        index_type q_idx;
        if (nw < delta){
            q_idx = atomicAdd(new_near_len,1);
            new_near_pile[q_idx] = far_idx;
        }
        else{
            q_idx = atomicAdd(new_far_len,1);
            new_far_pile[q_idx] = far_idx;
        }
        // printf("Adding %d to q[%d]\n",n,q_idx);
    }

}



void nf_impl(CSRGraph& g, edge_data_type* dists) {
    
    double start,end = 0;
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node = 0. Otherwise need to change this
    check_cuda(cudaMemset(&d_d[1], 0xFF,  (g.nnodes-1) * sizeof(edge_data_type)));

    index_type* near1 = NULL, *near2 = NULL, *far1 = NULL, *far2 = NULL;
    check_cuda(cudaMalloc(&near1, (g.nedges + 1) / 2 * sizeof(index_type)));
    check_cuda(cudaMalloc(&near2, (g.nedges + 1) / 2 * sizeof(index_type)));
    check_cuda(cudaMalloc(&far1, (g.nedges + 1) * sizeof(index_type)));
    check_cuda(cudaMalloc(&far2, (g.nedges + 1) * sizeof(index_type)));
    // Set first q entry to 0 (source) TODO: other sources
    check_cuda(cudaMemset(near1, 0, sizeof(index_type)));
    index_type* near_len = NULL, *far_len = NULL;
    check_cuda(cudaMallocManaged(&near_len, sizeof(index_type)));
    check_cuda(cudaMallocManaged(&far_len, sizeof(index_type)));
    *near_len = 1, *far_len = 1;


    printf("weight %d \n", g.getWeight(0,0)); 
    start = getTimeStamp();
    int iter = 0; 
    //todo: delta as warp size * average weight / average degree
    edge_data_type delta = 200; 
    while (*far_len > 0 || *near_len > 0) {
        printf("Iter %d, near batch %d, far batch %d, delta %d\n",iter, *near_len, *far_len, delta);
        index_type old_near_len = *near_len;
        *near_len = 0;
        // keep 
        nf_iter<<<(old_near_len + 512 - 1) / 512,512>>>(d_g, d_d, delta, near1, old_near_len, near2, near_len, far2, far_len);
        cudaDeviceSynchronize();
        
        printf("after update, near batch %d, far batch %d\n", *near_len, *far_len); 
        if (*near_len == 0){
            //todo: need to use a while loop to update delta until near batch has something
            //used up all near pile, time to update delta and split far pile into new near, far 
            delta += delta; 
            index_type old_far_len = *far_len;
            *far_len = 0; 
            *near_len = 0; 
            far_split<<<(old_far_len + 512 - 1) / 512,512>>>(d_d, delta, far2, old_far_len, near1, near_len, far1, far_len);
            cudaDeviceSynchronize();
            index_type* tmp = far1;
            far1 = far2;
            far2 = tmp; 
            printf("after delta_update, near batch %d, far batch %d , delta %d\n", *near_len, *far_len, delta); 
        }
        else{
            //continue working on near pile, switch near pile, keep adding to same far pile
            index_type* tmp = near1;
            near1 = near2;
            near2 = tmp; 
        }

    }
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);

}
