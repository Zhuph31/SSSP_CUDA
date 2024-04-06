#include "workfront-sweep.cuh"
#include <cub/device/device_select.cuh>
#include <cub/block/block_scan.cuh>


__global__ void wf_iter_aq(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx, index_type* scratch) {
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

            //printf("source %u, dst %u, new_w %u\n", s_idx, n, new_w);

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

template <int block_size>
void wf_sweep_atomicq(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source) {
    double start,end = 0;
    index_type* q1, *q2 = NULL;
    index_type* qscratch = NULL;
    check_cuda(cudaMalloc(&q1, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&q2, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&qscratch, g.nnodes* sizeof(index_type)));
    // Set first q entry to 0 (source) TODO: other sources
    check_cuda(cudaMemcpy(q1, &source, sizeof(index_type), cudaMemcpyHostToDevice));
    index_type* qlen = NULL;
    check_cuda(cudaMallocManaged(&qlen, sizeof(index_type)));
    *qlen = 1;

    //index_type* hq = NULL;
    //cudaHostAlloc(&hq,g.nnodes*sizeof(index_type),cudaHostAllocDefault);

    start = getTimeStamp();

    int itr = 0;
    while (*qlen) {
        printf("Iter %d, qlen %d\n",itr, *qlen);
        index_type len = *qlen;
        *qlen = 0;
 
        wf_iter_aq<<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_dists, q1, len,q2, qlen, qscratch);
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
}

#define THREADS_PER_BLOCK 256

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

    if (value >= block[hi - 1])
        return hi - 1; // if all elemenst in the scan result are smaller than value, the source vertex is the last one 

    index_type mid;
    while (lo < hi){
        mid = lo + (hi - lo) / 2; 
        if (block[mid] > value)
            hi = mid;
        else
            lo = mid + 1;
    }
    return (lo > 0) ? lo - 1 : lo;
    
    // for (int i = 0; i < hi - 1; i++){
    //     if (block[i] <= value && block[i + 1] > value)
    //         return i; 
    // }
    // return hi - 1; // if all elemenst in the scan result are smaller than value, the source vertex is the last one 
}



template <int block_size>
__global__ void wf_coop_iter_impl2(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* new_q, index_type* pq_idx,  index_type* scratch) {
    
    //one for start, one for end.
    __shared__ index_type source_vertices[block_size];
    __shared__ index_type offset_start[block_size];
    __shared__ index_type num_neighbors[block_size]; 

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
        for (int i = 1; i < block_size; i++){
            temp = num_neighbors[i];
            num_neighbors[i] = total_neighbors;
            total_neighbors += temp;
        }

        //printf("block_id %d, total neighbors per block %d \n", blockIdx.x, total_neighbors);
    }

    __syncthreads(); 
    
    /*********************************************************************************************************/
    //each take on a task
    for (index_type work_index = local_index; work_index < total_neighbors; work_index += block_size){

        //find shared mem index, so we can find source, offset start, and degree
        
        index_type shared_mem_index = bisect_right(num_neighbors, 0, block_size, work_index);
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




template <int block_size>
void wf_sweep_coop(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_d, index_type source) {
    double start,end = 0;
    index_type* q1, *q2 = NULL;
    index_type* qscratch = NULL;
    check_cuda(cudaMalloc(&q1, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&q2, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&qscratch, g.nnodes* sizeof(index_type)));
    // Set first q entry to source
    check_cuda(cudaMemcpy(q1, &source, sizeof(index_type), cudaMemcpyHostToDevice));
    index_type* qlen = NULL;
    check_cuda(cudaMallocManaged(&qlen, sizeof(index_type)));
    *qlen = 1;

    //index_type* hq = NULL;
    //cudaHostAlloc(&hq,g.nnodes*sizeof(index_type),cudaHostAllocDefault);

    start = getTimeStamp();

    int itr = 0;
    while (*qlen) {
        printf("Iter %d, qlen %d\n",itr, *qlen);
        index_type len = *qlen;
        *qlen = 0;
 
        //wf_coop_iter_impl1<<<(len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        wf_coop_iter_impl2<block_size><<<(len + block_size - 1) / block_size, block_size>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
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
}

///////////////////////////////////////////////////////////////////////////////

__global__ void wf_iter_filter(CSRGraph g, edge_data_type* d, index_type* last_q, index_type last_q_len, index_type* touched) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < last_q_len) {
        index_type s_idx = last_q[index];
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
                touched[n] = 1;
            }
        }
    }
}

__global__ void setup_id(index_type* out, index_type n) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < n)  out[index] = index;
}

void wf_sweep_filter(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source) {
    double start,end = 0;
    index_type* q = NULL;
    index_type* scan_indices = NULL;
    index_type* touched = NULL;
    check_cuda(cudaMalloc(&q, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&touched, g.nnodes* sizeof(index_type)));
    check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
    setup_id<<<(g.nnodes + 512 - 1),512>>>(scan_indices,g.nnodes);

    // Set first q entry to source
    check_cuda(cudaMemcpy(q, &source, sizeof(index_type),cudaMemcpyHostToDevice));
    index_type* qlen = NULL;
    check_cuda(cudaMallocManaged(&qlen, sizeof(index_type)));
    *qlen = 1;

    void* flg_tmp_store = NULL;
    size_t flg_store_size = 0;
    // index_type num_selected = 0;
    cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,q,qlen,g.nnodes);
    check_cuda(cudaMalloc(&flg_tmp_store,flg_store_size));


    start = getTimeStamp();
    while (*qlen) {
        printf("Iter %d\n",*qlen);
        index_type len = *qlen;
        *qlen = 0;
 
        wf_iter_filter<<<(len + 512 - 1) / 512,512>>>(d_g, d_dists, q, len, touched);
        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,q,qlen,g.nnodes);
        check_cuda(cudaMemset(touched,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

}

///////////////////////////////////////////////////////////////////////////////////////////

template <int block_size>
__global__ void wf_frontier_kernel(CSRGraph g, edge_data_type* d, index_type* frontier_in, index_type* frontier_out, index_type n, index_type* block_offsets) {
    __shared__ index_type vertices[block_size];
    __shared__ index_type first_edge_offset[block_size];
    __shared__ index_type output_offset[block_size];
    __shared__ uint64_t block_offset[1];

    // Specialize BlockScan for a 1D block of 128 threads on type int
    typedef cub::BlockScan<index_type, block_size> BlockScan;
     
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;

    index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);
    index_type tidx = threadIdx.x;
    
    index_type degree = 0;
    if (gidx < n) {
        index_type v = frontier_in[gidx];
        if (v != UINT_MAX) {
            vertices[tidx] = v;
            // if (helper[v]) {
            //     printf("BAD! Vertex %d already grabbed!\n",v);
            // }
            // helper[v] = 1;
            index_type row_start =  g.row_start[v];
            first_edge_offset[tidx] = row_start;
            degree = g.row_start[v+1] - row_start;
        } else {
            vertices[tidx] = UINT_MAX;
        }
    }
    // if (gidx == 0) {
    // printf("Thread %d has node %d with degree %d\n",tidx, vertices[tidx], degree);
    // }

    __syncthreads();

    index_type block_aggregate = 0;
    BlockScan(temp_storage).ExclusiveSum(degree, degree, block_aggregate);
    output_offset[tidx] = degree;
    // if (gidx == 0) {
    // printf("Block aggregate %d\n",block_aggregate);
    // }
    __syncthreads();

    if (tidx == 0 && block_aggregate) {
        block_offset[0] = atomicAdd(block_offsets,block_aggregate);
    }
    // if (gidx == 0) {
    // printf("\nBlock totals %d\n",*block_offsets);
    // }


    __syncthreads();

    for (index_type edge_id = tidx; edge_id < block_aggregate; edge_id += block_size) {
        // search for edge
        index_type v_id = 0;
        {
            index_type lo = 0;
            index_type hi = block_size;
            while (lo != hi-1) {
                v_id = lo + (hi - lo)/2;
                if (edge_id >= output_offset[v_id]) {
                    lo = v_id;
                } else {
                    hi = v_id;
                }
            }
            v_id = lo;
        }

        index_type edge_offset = edge_id - output_offset[v_id];
        index_type v_in = vertices[v_id];
        // if (v_in == 0) printf("Got vertex 0! Edge_id %d\n", edge_id);
        index_type edge_dst = g.edge_dst[first_edge_offset[v_id]+ edge_offset];
        edge_data_type ew = g.edge_data[first_edge_offset[v_id]+ edge_offset];
        // printf("exploring edge %d\n",first_edge_offset[v_id]+ edge_offset);
        edge_data_type vw = d[v_in];
        edge_data_type old_dw = d[edge_dst];
        edge_data_type new_dw = vw + ew;

        index_type out_val = UINT_MAX;
        if (new_dw < old_dw) {
            atomicMin(&d[edge_dst], new_dw);
            out_val = edge_dst;
        }
        frontier_out[block_offset[0] + edge_id] = out_val;
        // } else {
        //     frontier_out[block_offset[0] + edge_id] = UINT_MAX;
        // }
        // if (d[edge_dst] == 0) printf("\nEdge ID %d. Using local vert %d (%d)source weight: %d. Dest: %d (%d). new: %d \n", edge_id, v_id, v_in, vw, edge_dst, old_dw, new_dw);

    }
}

__global__ void filter_frontier(index_type* frontier_in, index_type* frontier_out, index_type n, index_type* visited, index_type iteration) {
    index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (gidx < n) {
        index_type v = frontier_in[gidx];
        index_type out = UINT_MAX;
        if (v != UINT_MAX) {
            if (atomicExch(&visited[v],iteration) != iteration) {
                out = v;
            } else {
                // printf("Skipping vertex %d\n",v);
            }
        }
        frontier_out[gidx] = out;
    }
}

template <int block_size>
void wf_sweep_frontier(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source) {
    double start,end = 0;
    index_type* frontier1, *frontier2 = NULL;
    check_cuda(cudaMalloc(&frontier1, g.nedges * sizeof(index_type)));
    check_cuda(cudaMalloc(&frontier2, g.nedges * sizeof(index_type)));
    index_type* m_N = NULL;
    index_type* visited = NULL;
    check_cuda(cudaMalloc(&visited, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMallocManaged(&m_N, sizeof(index_type)));
    *m_N = 1;
    check_cuda(cudaMemcpy(frontier1,&source, sizeof(index_type), cudaMemcpyHostToDevice));


    start = getTimeStamp();

    index_type iter = 0;
    while(*m_N) {
        index_type n = *m_N;
        *m_N = 0;
        printf("Iter %d\n",n);

        wf_frontier_kernel<block_size><<<(n + block_size-1)/block_size,block_size>>>(d_g, d_dists, frontier1, frontier2, n, m_N);
        cudaDeviceSynchronize();


        // filter
        n = *m_N;
        filter_frontier<<<(n + block_size-1)/block_size,block_size>>>(frontier2, frontier1,n,visited,iter+1);
        cudaDeviceSynchronize();

        iter++;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);
}



/////////////////////////////////////////////////////////



void workfront_sweep(CSRGraph& g, edge_data_type* dists, index_type source) {
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node = 0. Otherwise need to change this
    check_cuda(cudaMemset(&d_d[0], 0xFF,  (g.nnodes) * sizeof(edge_data_type)));
    check_cuda(cudaMemset(&d_d[source], 0,  sizeof(edge_data_type)));

    wf_sweep_frontier<256>(g, d_g, d_d, source);    


    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}
