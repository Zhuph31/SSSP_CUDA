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

            printf("source %u, dst %u, new_w %u\n", s_idx, n, new_w);

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



__device__ index_type bisect_right(index_type *block, index_type lo, index_type hi, index_type value){
    // index_type mid;
    // while (lo < hi){
    //     mid = lo + (hi - lo) / 2; 
    //     if (block[mid] > value)
    //         hi = mid;
    //     else
    //         lo = mid + 1;

    //     return (lo > 0) ? lo - 1 : lo;

    // }

    for (int i = 0; i < hi - 1; i++){
        if (block[i] <= value && block[i + 1] > value)
            return i; 
    }
    return hi - 1; // if all elemenst in the scan result are smaller than value, the source vertex is the last one 
}





#define THREAD_PER_BLOCK 512
__global__ void wf_coop_iter_impl2(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx,  index_type* scratch) {
    
    //one for start, one for end.
    __shared__ index_type source_vertices[THREAD_PER_BLOCK];
    __shared__ index_type offset_start[THREAD_PER_BLOCK];
    __shared__ index_type num_neighbors[THREAD_PER_BLOCK]; 

    __shared__ index_type total_neighbors;



    index_type global_index = threadIdx.x + (blockDim.x * blockIdx.x);
    index_type local_index = threadIdx.x; 


    //initialize source vertices, else bisect wont work properly
    source_vertices[local_index] = 0; 
    num_neighbors[local_index] = 0;
    //offset_start[local_index] = 0;
    

    //decide what each block work range is 
    index_type block_index_start = blockIdx.x * blockDim.x;
    index_type block_index_end = block_index_start + blockDim.x;  
    //do not go beyond last_q_len
    block_index_end = min(block_index_end, last_q_len); 

    if (global_index < block_index_end){
        //load start and end offset and number of neighbors into shared memory
        index_type s_idx = last_q[global_index];
        source_vertices[local_index] = s_idx; 
        offset_start[local_index] = g.row_start[s_idx];
        num_neighbors[local_index] = g.row_start[s_idx + 1] - g.row_start[s_idx]; 

        //printf("block_idx %d, global_index %d, block_end %d, num_neighbors %d, source %d\n", blockIdx.x, global_index, block_index_end, num_neighbors[local_index], s_idx);
        
    }
    __syncthreads();

    /****************************** Replace with scan ********************************************************/
    //add total num_neighbors to determine total work, replace with block level exclusive scan also get sum
    if (local_index == 0){
        total_neighbors = num_neighbors[0];
        num_neighbors[0] = 0; 
        index_type temp = 0; 
        for (int i = 1; i < THREAD_PER_BLOCK; i++){
            temp = num_neighbors[i];
            num_neighbors[i] = total_neighbors;
            total_neighbors += temp;
        }

        //printf("block_id %d, total neighbors per block %d \n", blockIdx.x, total_neighbors);
    }

    __syncthreads(); 
    
    /*********************************************************************************************************/
    //each take on a task
    for (index_type work_index = local_index; work_index < total_neighbors; work_index += THREAD_PER_BLOCK){

        //find shared mem index, so we can find source, offset start, and degree
        
        index_type shared_mem_index = bisect_right(num_neighbors, 0, THREAD_PER_BLOCK, work_index);
        index_type source = source_vertices[shared_mem_index];
        index_type edge_index = offset_start[shared_mem_index] + work_index - num_neighbors[shared_mem_index];

        
        //rest of code remains the same 
        edge_data_type w = d[source];
        edge_data_type ew = g.edge_data[edge_index];
        index_type n = g.edge_dst[edge_index];
        edge_data_type nw = d[n];
        edge_data_type new_w = ew + w;

        // Check if the distance is already set to max then just take the max since,
        if (w >= MAX_VAL){
            new_w = MAX_VAL;
        }

       //printf("local_index %u, worker_index %u, shared_mem_index %u, source %u, dst %d, new_w %d\n", local_index, work_index, shared_mem_index, source, n, new_w);

        if (new_w < nw) {
            atomicMin(&d[n],new_w);
            if (atomicCAS(&scratch[n],0,1) == 0) {
                index_type q_idx = atomicAdd(pq_idx,1);
                new_q[q_idx] = n;
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

    int itr = 0;
    while (*qlen) {
        printf("Iter %d\n",*qlen);
        index_type len = *qlen;
        *qlen = 0;
 
        //wf_iter<<<(len + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        //wf_coop_iter_impl1<<<(len + 128 - 1) / 128,128>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        wf_coop_iter_impl2<<<(len + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        check_cuda(cudaMemset(qscratch,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = q1;
        q1 = q2;
        q2 = tmp;


        itr += 1;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}
