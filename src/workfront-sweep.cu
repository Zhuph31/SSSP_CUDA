#include "workfront-sweep.cuh"


__global__ void wf_iter(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx, index_type* scratch) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < last_q_len) {
        index_type s_idx = last_q[index];
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
                if (atomicCAS(&scratch[n],0,index) == 0) {
                    index_type q_idx = atomicAdd(pq_idx,1);
                    new_q[q_idx] = n;
                }
            }
        }
    }
}

void workfront_sweep(CSRGraph& g, edge_data_type* dists) {
    double start,end = 0;
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node = 0. Otherwise need to change this
    check_cuda(cudaMemset(&d_d[1], 0xFF,  (g.nnodes-1) * sizeof(edge_data_type)));

    index_type* q1, *q2 = NULL;
    index_type* qscratch = NULL;
    check_cuda(cudaMalloc(&q1, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&q2, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&qscratch, g.nnodes* sizeof(index_type)));
    // Set first q entry to 0 (source) TODO: other sources
    check_cuda(cudaMemset(q1, 0, sizeof(index_type)));
    index_type* qlen = NULL;
    check_cuda(cudaMallocManaged(&qlen, sizeof(index_type)));
    *qlen = 1;

    //index_type* hq = NULL;
    //cudaHostAlloc(&hq,g.nnodes*sizeof(index_type),cudaHostAllocDefault);

    start = getTimeStamp();
    while (*qlen) {
        printf("Iter %d\n",*qlen);
        index_type len = *qlen;
        *qlen = 0;
 
        wf_iter<<<(len + 512 - 1) / 512,512>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        check_cuda(cudaMemset(qscratch,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = q1;
        q1 = q2;
        q2 = tmp;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}
