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

__global__ void wf_coop_iter_impl1(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx,  index_type* scratch) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    __shared__ index_type block_row_start;
    __shared__ index_type block_row_end; 
    __shared__ index_type s_idx; 

    //decide what each block work range is 
    index_type block_index_start = blockIdx.x * blockDim.x;
    index_type block_index_end = block_index_start + blockDim.x;  
    //do not go beyond last_q_len
    block_index_end = min(block_index_end, last_q_len); 

    for (index_type index=block_index_start; index < block_index_end; index++){
        //only first thread in block load row start and end and source index 
        if (threadIdx.x == 0){
            s_idx = last_q[index];
            block_row_start = g.row_start[s_idx]; 
            block_row_end = g.row_start[s_idx + 1]; 
            //printf("source id %d, block_row_start %d, block_row_end %d \n", s_idx, block_row_start, block_row_end);
        }
        __syncthreads();

        //the threads in this block each take one edge
        for (index_type j =threadIdx.x + block_row_start; j < block_row_end; j += blockDim.x){
            
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


__device__ index_type bisect_left(index_type *block, index_type lo, index_type hi, index_type value){
    index_type mid;
    while (lo < hi){
        mid = lo + (hi - lo) / 2; 
        if (block[mid] < value)
            lo = mid + 1;
        else
            hi = mid; 
    }
    
    return (lo > 0) ? lo - 1 : lo; 
}





#define BLOCK_DIM_X 128
__global__ void wf_coop_iter_impl2(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx,  index_type* scratch) {
    
    //one for start, one for end.
    __shared__ index_type block_row_offset[BLOCK_DIM_X * 2];
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    //decide what each block work range is 
    index_type block_index_start = blockIdx.x * blockDim.x;
    index_type block_index_end = block_index_start + blockDim.x;  
    index_type s_idx; 
    //do not go beyond last_q_len
    block_index_end = min(block_index_end, last_q_len); 
    __shared__ int max_index; 
    max_index = -1;

    if (index < block_index_end){
        //load start and end offset into shared memory
        s_idx = last_q[index];
        block_row_offset[threadIdx.x * 2] = g.row_start[s_idx];
        block_row_offset[threadIdx.x * 2 + 1] = g.row_start[s_idx + 1]; 
        atomicAdd(&max_index, 2); 
        printf("index %d, s_idx %d, start %d, end %d, max_index %d\n", index ,s_idx, g.row_start[s_idx], g.row_start[s_idx + 1], max_index);
    }
    __syncthreads();


    index_type row_offset = threadIdx.x + block_row_offset[0];
    index_type block_max = block_row_offset[max_index];

    //all thread work on one item in the offset array 
    //example offset_array 0, 2, 8, 16, 19, a block have 8 threads
    //first work on 0, 2, 8, then work on 8, 16, last work on 16, 19
    while (row_offset < block_max){

        //find source index
        s_idx = bisect_left(block_row_offset, 0, max_index, row_offset);

        edge_data_type w = d[s_idx];
        edge_data_type ew = g.edge_data[row_offset];
        index_type n = g.edge_dst[row_offset];
        edge_data_type nw = d[n];
        edge_data_type new_w = ew + w;

        printf("working, row_offset %d, s_idx %d, d_idx %d, new_w %d, edge weight %d, weight %d\n", row_offset, s_idx, n, new_w, ew, w); 
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

        row_offset += blockDim.x; 
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
 
        //wf_iter<<<(len + 512 - 1) / 512,512>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        wf_coop_iter_impl1<<<(len + 128 - 1) / 128,128>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        //wf_coop_iter_impl2<<<(len + 128 - 1) / 128,128>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
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
