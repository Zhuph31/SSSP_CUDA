#include "workfront-sweep.cuh"
#include <cub/device/device_select.cuh>
#include <cub/block/block_scan.cuh>

enum OutType {
    QUEUE,
    FILTER_COMPACT,
    FILTER_IGNORE,
    FRONTIER
};


enum CoopType{
    VANILLA,
    FULL
};

struct TimeCost{
  double gpu_time;
  double overhead;
  TimeCost () = default;
  TimeCost(double g, double o) : gpu_time(g), overhead(o) {}
};

template <OutType out_type>
__global__ void wf_iter_simple(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type input_len, index_type* output, index_type* output_len, index_type* vertex_claim) {
    index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (gidx < input_len) {
        // Take a vertex from the work list
        index_type source_vtx = worklist_in[gidx];
        edge_data_type source_cost = best_costs[source_vtx];

        // Process each of the vertex's node's
        for (int e = g.row_start[source_vtx]; e < g.row_start[source_vtx + 1]; e++) {
            edge_data_type edge_weight = g.edge_data[e];
            index_type dest_vtx = g.edge_dst[e];
            edge_data_type old_dest_cost = best_costs[dest_vtx];
            edge_data_type new_dest_cost = edge_weight + source_cost;

            if (new_dest_cost < old_dest_cost) {
                // Update improved cost in global memory
                atomicMin(&best_costs[dest_vtx],new_dest_cost);

                // Add the destination vertex to next iteration's work list
                if (out_type == OutType::QUEUE) {
                    if (atomicCAS(&vertex_claim[dest_vtx],0,gidx) == 0) {
                        index_type q_idx = atomicAdd(output_len,1);
                        output[q_idx] = dest_vtx;
                    }
                } else if (out_type == OutType::FILTER_COMPACT) {
                    output[dest_vtx] = 1;
                }
            }
        }
    }
}

template <int block_size>
TimeCost wf_sweep_atomicq(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* q1, *q2 = NULL;
    index_type* qscratch = NULL;


    double setup_start = getTimeStamp();
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
    overhead += start - setup_start;

    int itr = 0;
    while (*qlen) {
        if (verbose) {
            printf("Iter %d, qlen %d\n",itr, *qlen);
        }
        index_type len = *qlen;
        *qlen = 0;
 
        wf_iter_simple<OutType::QUEUE><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_dists, q1, len,q2, qlen, qscratch);
        check_cuda(cudaMemset(qscratch,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = q1;
        q1 = q2;
        q2 = tmp;


        itr += 1;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(q1));
    check_cuda(cudaFree(q2));
    check_cuda(cudaFree(qscratch));
    check_cuda(cudaFree(qlen));
    overhead += getTimeStamp() - end;
    return {gpu_time, overhead};
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <int block_size, OutType out_type>
__global__ void wf_coop_iter_impl1(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type input_len, index_type* output, index_type* output_len,  index_type* vertex_claim) {

    __shared__ index_type block_row_start;
    __shared__ index_type block_row_end; 
    // __shared__ index_type source_vtx;
    __shared__ index_type source_cost_shared;

    // decide what each block's work range is 
    index_type block_index_start = blockIdx.x * blockDim.x;
    index_type block_index_end = block_index_start + blockDim.x;  
    // do not go beyond input_len
    block_index_end = min(block_index_end, input_len);

    for (index_type index=block_index_start; index < block_index_end; index++){
        //only first thread in block load row start and end and source index 
        if (threadIdx.x == 0){
            index_type source_vtx = worklist_in[index];
            block_row_start = g.row_start[source_vtx]; 
            block_row_end = g.row_start[source_vtx + 1]; 
            source_cost_shared = best_costs[source_vtx];
        }
        __syncthreads();

        edge_data_type source_cost = source_cost_shared;
        //the threads in this block each take one edge
        for (index_type e = threadIdx.x + block_row_start; e < block_row_end; e += blockDim.x){
            edge_data_type edge_weight = g.edge_data[e];
            index_type dest_vtx = g.edge_dst[e];
            edge_data_type old_dest_cost = best_costs[dest_vtx];
            edge_data_type new_dest_cost = edge_weight + source_cost;

            if (new_dest_cost < old_dest_cost) {
                // Update improved cost in global memory
                atomicMin(&best_costs[dest_vtx],new_dest_cost);

                // Add the destination vertex to next iteration's work list
                if (out_type == OutType::QUEUE) {
                    if (atomicCAS(&vertex_claim[dest_vtx],0,1) == 0) {
                        index_type q_idx = atomicAdd(output_len,1);
                        output[q_idx] = dest_vtx;
                    }
                } else if (out_type == OutType::FILTER_COMPACT) {
                    output[dest_vtx] = 1;
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
}



template <int block_size, OutType out_type>
__global__ void wf_coop_iter_impl2(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type input_len, index_type* output, index_type* outpu_len,  index_type* vertex_claim, index_type iter) {
    
    //one for start, one for end.
    __shared__ index_type source_vertices[block_size];
    __shared__ index_type offset_start[block_size];
    __shared__ index_type num_neighbors[block_size]; 

    __shared__ index_type total_neighbors;

    __shared__ index_type frontier_output_start[1];

    // Specialize BlockScan for a 1D block of block_size threads on type int
    typedef cub::BlockScan<index_type, block_size> BlockScan;
     
    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;


    index_type global_index = threadIdx.x + (blockDim.x * blockIdx.x);
    index_type local_index = threadIdx.x; 


    //initialize source vertices, else bisect wont work properly
    source_vertices[local_index] = 0; 
    // num_neighbors[local_index] = 0;
    //offset_start[local_index] = 0;
    

    //decide what each block work range is 
    index_type block_index_start = blockIdx.x * blockDim.x;
    index_type block_index_end = block_index_start + blockDim.x;  
    //do not go beyond last_q_len
    block_index_end = min(block_index_end, input_len); 


    index_type num_neighbors_local = 0;
    if (global_index < block_index_end){
        //load start and end offset and number of neighbors into shared memory
        if (out_type == FILTER_IGNORE) {
            // Filter ignore: input is last iteration each vertex was updated
            index_type s_idx = global_index;
            if (worklist_in[s_idx] == iter -1 ) {
                // printf("Found vertex %d\n",s_idx);
                source_vertices[local_index] = s_idx;
                index_type row_start = g.row_start[s_idx];
                num_neighbors_local = g.row_start[s_idx + 1] - row_start;
                offset_start[local_index] = row_start;
            }
        } else if (out_type == FRONTIER) {
             // Frontier: input is a source vertex ID or UINT_MAX
            index_type v = worklist_in[global_index];
            if (v != UINT_MAX ) {
                // printf("Found vertex %d\n",s_idx);
                source_vertices[local_index] = v;
                index_type row_start = g.row_start[v];
                num_neighbors_local = g.row_start[v + 1] - row_start;
                offset_start[local_index] = row_start;
            }
        } else {
            // Input is list of vertex id's to check
            index_type s_idx = worklist_in[global_index];
            source_vertices[local_index] = s_idx; 
            index_type row_start = g.row_start[s_idx];
            num_neighbors_local = g.row_start[s_idx + 1] - row_start;
            offset_start[local_index] = row_start;
        }

        //printf("block_idx %d, global_index %d, block_end %d, num_neighbors %d, source %d\n", blockIdx.x, global_index, block_index_end, num_neighbors[local_index], s_idx);
        
    }
    __syncthreads();

    /****************************** Replace with scan ********************************************************/
    //add total num_neighbors to determine total work, replace with block level exclusive scan also get sum
        // __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(num_neighbors_local, num_neighbors_local, total_neighbors);
    // output_offset[tidx] = degree;
    // if (global_index == 0) {
    // printf("Block aggregate %d\n",total_neighbors);
    // }
    if (total_neighbors == 0) {
        return;
    }
    num_neighbors[local_index] = num_neighbors_local;
    // __syncthreads();
    
    if (local_index == 0 && (out_type == FILTER_IGNORE || out_type == FRONTIER)) {
        frontier_output_start[0] = atomicAdd(outpu_len,total_neighbors);
    }



    // if (local_index == 0){
    //     total_neighbors = num_neighbors[0];
    //     num_neighbors[0] = 0; 
    //     index_type temp = 0; 
    //     for (int i = 1; i < block_size; i++){
    //         temp = num_neighbors[i];
    //         num_neighbors[i] = total_neighbors;
    //         total_neighbors += temp;
    //     }

    //     //printf("block_id %d, total neighbors per block %d \n", blockIdx.x, total_neighbors);
    // }

    __syncthreads(); 
    
    /*********************************************************************************************************/
    //each take on a task
    for (index_type work_index = local_index; work_index < total_neighbors; work_index += block_size){

        //find shared mem index, so we can find source, offset start, and degree
        
        index_type shared_mem_index = bisect_right(num_neighbors, 0, block_size, work_index);
        index_type source = source_vertices[shared_mem_index];
        index_type edge_index = offset_start[shared_mem_index] + work_index - num_neighbors[shared_mem_index];

        
        //rest of code remains the same 
        edge_data_type w = best_costs[source];
        edge_data_type ew = g.edge_data[edge_index];
        index_type n = g.edge_dst[edge_index];
        edge_data_type nw = best_costs[n];
        edge_data_type new_w = ew + w;

        // Check if the distance is already set to max then just take the max since,
        if (w >= MAX_VAL){
            new_w = MAX_VAL;
        }

       //printf("local_index %u, worker_index %u, shared_mem_index %u, source %u, dst %d, new_w %d\n", local_index, work_index, shared_mem_index, source, n, new_w);

        if (out_type != OutType::FRONTIER) {
            if (new_w < nw) {
                atomicMin(&best_costs[n],new_w);
                if (out_type == OutType::QUEUE){
                    if (atomicCAS(&vertex_claim[n],0,1) == 0) {
                        index_type q_idx = atomicAdd(outpu_len,1);
                        output[q_idx] = n;
                    }
                }
                else if (out_type == OutType::FILTER_IGNORE){
                    output[n] = iter;
                } else if (out_type == OutType::FILTER_COMPACT) {
                    // printf("Flagging %d\n",n);
                    output[n] = 1;
                }
            }
        } else {
            index_type out_val = UINT_MAX;
            if (new_w < nw) {
                atomicMin(&best_costs[n], new_w);
                out_val = n;
            }
            // printf("writing %d to output %d(source %d)\n",out_val, frontier_output_start[0] + work_index, source);
            output[frontier_output_start[0] + work_index] = out_val;
        }

         
    }
    
}


template <int block_size, CoopType coop_impl>
TimeCost wf_sweep_coop(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_d, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* q1, *q2 = NULL;
    index_type* qscratch = NULL;
    double setup_start = getTimeStamp();
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
    overhead += start - setup_start;

    int itr = 0;
    while (*qlen) {
        if (verbose) {
            printf("Iter %d, qlen %d\n",itr, *qlen);
        }
        index_type len = *qlen;
        *qlen = 0;

        if (coop_impl == CoopType::VANILLA)
            wf_coop_iter_impl1<block_size, OutType::QUEUE><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_d, q1, len,q2, qlen, qscratch);
        else if (coop_impl == CoopType::FULL)
            wf_coop_iter_impl2<block_size, OutType::QUEUE><<<(len + block_size - 1) / block_size, block_size>>>(d_g, d_d, q1, len,q2, qlen, qscratch,0);
        check_cuda(cudaMemset(qscratch,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = q1;
        q1 = q2;
        q2 = tmp;


        itr += 1;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(q1));
    check_cuda(cudaFree(q2));
    check_cuda(cudaFree(qscratch));
    check_cuda(cudaFree(qlen));

    overhead += getTimeStamp() - end;

    return {gpu_time, overhead};
}

__global__ void setup_id(index_type* out, index_type n) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < n)  out[index] = index;
}

__global__ void filter_frontier(index_type* frontier_in, index_type* frontier_out, index_type frontier_len, index_type* visited, index_type iteration) {
    index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (gidx < frontier_len) {
        index_type v = frontier_in[gidx];
        index_type out = UINT_MAX;
        if (v != UINT_MAX) {
            if (atomicExch(&visited[v],iteration) != iteration) {
                out = v;
            }
        }
        frontier_out[gidx] = out;
    }
}


template <int block_size, CoopType coop_impl, OutType out_type>
TimeCost wf_coop_filter(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* q = NULL;
    index_type* scan_indices = NULL;
    index_type* touched = NULL;
    double setup_start = getTimeStamp();
    if (out_type != OutType::FRONTIER) {
        check_cuda(cudaMalloc(&q, g.nnodes * sizeof(index_type)));
        check_cuda(cudaMalloc(&touched, g.nnodes* sizeof(index_type)));
    } else {
        check_cuda(cudaMalloc(&q, g.nedges * sizeof(index_type)));
        check_cuda(cudaMalloc(&touched, g.nedges* sizeof(index_type)));
    }

    if (out_type == OutType::FILTER_COMPACT || out_type == OutType::FRONTIER) {
        check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
    }
    if (out_type == OutType::FILTER_COMPACT) {
    setup_id<<<(g.nnodes + block_size - 1),block_size>>>(scan_indices,g.nnodes);
    } else if (out_type == OutType::FRONTIER) {
    check_cuda(cudaMemset(scan_indices,0,g.nnodes * sizeof(index_type)));
    }

    // Set first q entry to source
    check_cuda(cudaMemcpy(q, &source, sizeof(index_type),cudaMemcpyHostToDevice));
    index_type* qlen = NULL;
    check_cuda(cudaMallocManaged(&qlen, sizeof(index_type)));
    *qlen = 1;

    void* flg_tmp_store = NULL;
    size_t flg_store_size = 0;
    // index_type num_selected = 0;
    if (out_type == OutType::FILTER_COMPACT) {
        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,q,qlen,g.nnodes);
        check_cuda(cudaMalloc(&flg_tmp_store,flg_store_size));
        cudaMemset(&touched[0],0,(g.nnodes)*sizeof(index_type));
    } else {
        cudaMemset(&q[1],0xFF,(g.nnodes-1)*sizeof(index_type));
        cudaMemset(&touched[0],0xFF,(g.nnodes)*sizeof(index_type));
    }

    start = getTimeStamp();
    overhead += start - setup_start;
    int iter = 0;
    while (*qlen) {
        if (verbose) {
            printf("Iter %d\n",*qlen);
        }
        index_type len = *qlen;
        *qlen = 0;

        // if (coop_impl == CoopType::VANILLA)
        //     wf_coop_iter_impl1<block_size, OutType::FILTER_IGNORE><<<(len + block_size - 1) / block_size, block_size>>>(d_g, d_dists, q, len, touched,qlen, NULL);
        // else if (coop_impl == CoopType::FULL)

        int n = (out_type == OutType::FILTER_IGNORE)?g.nnodes:len;
        wf_coop_iter_impl2<block_size, out_type>
            <<<(n + block_size - 1) / block_size, block_size>>>
            (d_g, d_dists, q, n, touched,qlen, NULL,iter+1);

        if (out_type == OutType::FILTER_COMPACT) {
            cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,q,qlen,g.nnodes);
            cudaDeviceSynchronize();
            check_cuda(cudaMemset(touched,0,g.nnodes*sizeof(index_type)));
        } else if (out_type == OutType::FILTER_IGNORE) {
            cudaDeviceSynchronize();
            index_type* tmp = q;
            q = touched;
            touched = tmp;
        } else if (out_type == OutType::FRONTIER) {
            check_cuda(cudaDeviceSynchronize());
            n = *qlen;
            filter_frontier<<<(n + block_size-1)/block_size,block_size>>>(touched, q,n,scan_indices,iter+1);
            check_cuda(cudaDeviceSynchronize());
        }
        iter++;

        // printf("res: %d\n",*qlen);
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(q));
    check_cuda(cudaFree(touched));
    check_cuda(cudaFree(scan_indices));
    check_cuda(cudaFree(qlen));
    overhead += getTimeStamp() - end;
    return {gpu_time, overhead};
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////



template <int block_size>
TimeCost wf_sweep_filter(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* q = NULL;
    index_type* scan_indices = NULL;
    index_type* touched = NULL;
    double setup_start = getTimeStamp();
    check_cuda(cudaMalloc(&q, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&touched, g.nnodes* sizeof(index_type)));
    check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
    setup_id<<<(g.nnodes + block_size - 1),block_size>>>(scan_indices,g.nnodes);

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
    overhead += start - setup_start;
    while (*qlen) {
        if (verbose) {
            printf("Iter %d\n",*qlen);
        }
        index_type len = *qlen;
        *qlen = 0;
 
        wf_iter_simple<OutType::FILTER_COMPACT><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_dists, q, len, touched,NULL, NULL);
        //coop example, interface exactly the same 
        //wf_coop_iter_impl2<block_size, OutType::TOUCHED><<<(len + block_size - 1) / block_size, block_size>>>(d_g, d_dists, q, len, touched,NULL, NULL);
        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,q,qlen,g.nnodes);
        check_cuda(cudaMemset(touched,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(q));
    check_cuda(cudaFree(touched));
    check_cuda(cudaFree(scan_indices));
    check_cuda(cudaFree(qlen));
    overhead += getTimeStamp() - end;
    return {gpu_time, overhead};
}

///////////////////////////////////////////////////////////////////////////////////////////

template <int block_size, OutType out_type>
__global__ void wf_frontier_kernel(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type* output, index_type input_len, index_type* output_len, index_type* vertex_claim, index_type iter) {
    __shared__ index_type vertex_costs[block_size];
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
    if (gidx < input_len) {
        if (out_type == OutType::FRONTIER) {
            index_type v = worklist_in[gidx];
            if (v != UINT_MAX) {
                index_type row_start =  g.row_start[v];
                degree = g.row_start[v+1] - row_start;
                first_edge_offset[tidx] = row_start;
                vertex_costs[tidx] = best_costs[v];
            }
        } else if (out_type == OutType::FILTER_IGNORE){
            index_type v = gidx;
            if (worklist_in[v] == iter - 1) {
                index_type row_start =  g.row_start[v];
                degree = g.row_start[v+1] - row_start;
                first_edge_offset[tidx] = row_start;
                vertex_costs[tidx] = best_costs[v];
            } else {
                v = UINT_MAX;
            }
        } else {
            index_type v = worklist_in[gidx];
            index_type row_start =  g.row_start[v];
            degree = g.row_start[v+1] - row_start;
            first_edge_offset[tidx] = row_start;
            vertex_costs[tidx] = best_costs[v];
        }
    }

    // if (gidx == 0) {
    // printf("Thread %d has node %d with degree %d\n",tidx, vertices[tidx], degree);
    // }
    

    index_type block_aggregate = 0;
    BlockScan(temp_storage).ExclusiveSum(degree, degree, block_aggregate);
    output_offset[tidx] = degree;
    // if (gidx == 0) {
    // printf("Block aggregate %d\n",block_aggregate);
    // }
    // __syncthreads();
    if (block_aggregate == 0) {
        return;
    }

    if (out_type == OutType::FRONTIER || out_type == OutType::FILTER_IGNORE) {
        if (tidx == 0 && block_aggregate) {
            // printf("adding %d to %d\n",block_aggregate, *out_size);
            block_offset[0] = atomicAdd(output_len,block_aggregate);
        }
    }
    // if (gidx == 0) {
    // printf("\nBlock totals %d\n",*out_size);
    // }


    __syncthreads();

    index_type last_vid = 0;
    for (index_type edge_id = tidx; edge_id < block_aggregate; edge_id += block_size) {
        // search for edge
        index_type v_id = last_vid;
        {
            index_type lo = last_vid;
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
            last_vid = v_id;
        }

        index_type edge_offset = edge_id - output_offset[v_id];
        // index_type source_vtx = vertices[v_id];
        index_type dest_vtx = g.edge_dst[first_edge_offset[v_id]+ edge_offset];
        edge_data_type edge_weight = g.edge_data[first_edge_offset[v_id]+ edge_offset];
        // printf("exploring edge %d\n",first_edge_offset[v_id]+ edge_offset);
        edge_data_type source_cost = vertex_costs[v_id];
        edge_data_type old_dest_cost = best_costs[dest_vtx];
        edge_data_type new_dest_cost = source_cost + edge_weight;

        if (out_type == OutType::FRONTIER) {
            index_type out_val = UINT_MAX;
            if (new_dest_cost < old_dest_cost) {
                atomicMin(&best_costs[dest_vtx], new_dest_cost);
                out_val = dest_vtx;
            }
            output[block_offset[0] + edge_id] = out_val;
        } else if (out_type == OutType::FILTER_IGNORE) {
            if (new_dest_cost < old_dest_cost) {
                atomicMin(&best_costs[dest_vtx], new_dest_cost);           
                output[dest_vtx] = iter;
            }
        } else if (out_type == OutType::FILTER_COMPACT) {
            if (new_dest_cost < old_dest_cost) {
                atomicMin(&best_costs[dest_vtx], new_dest_cost);           
                output[dest_vtx] = 1;
            }
        } else if (out_type == OutType::QUEUE) {
            if (new_dest_cost < old_dest_cost) {
                atomicMin(&best_costs[dest_vtx], new_dest_cost);
                if (atomicCAS(&vertex_claim[dest_vtx],0,1) == 0) {
                    index_type q_idx = atomicAdd(output_len,1);
                    output[q_idx] = dest_vtx;
                }
            }
        }

    }
}



template <int block_size, OutType out_type>
TimeCost wf_sweep_frontier(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* list1, *list2, *vertex_claim = NULL;
    double setup_start = getTimeStamp();
    check_cuda(cudaMalloc(&list1, g.nedges * sizeof(index_type)));
    check_cuda(cudaMalloc(&list2, g.nedges * sizeof(index_type)));
    index_type* m_N = NULL;
    check_cuda(cudaMallocManaged(&m_N, sizeof(index_type)));
    *m_N = 1;
    check_cuda(cudaMemcpy(list1,&source, sizeof(index_type), cudaMemcpyHostToDevice));

    index_type* scan_indices = NULL;
    void* flg_tmp_store = NULL;
    size_t flg_store_size = 0;
    if (out_type == OutType::FILTER_COMPACT) {
        check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
        setup_id<<<(g.nnodes + block_size - 1),block_size>>>(scan_indices,g.nnodes);
        // index_type num_selected = 0;
        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,list1,list2,m_N,g.nnodes);
        check_cuda(cudaMalloc(&flg_tmp_store,flg_store_size));
    }

    if (out_type == OutType::FILTER_IGNORE) {
        cudaMemset(&list1[1], 0xFF, (g.nnodes-1)*sizeof(index_type));
        cudaMemset(&list2[0], 0xFF, (g.nnodes)*sizeof(index_type));
    }

    if (out_type == OutType::QUEUE || out_type == OutType::FRONTIER) {
        cudaMalloc(&vertex_claim, g.nnodes * sizeof(index_type));
    }

    start = getTimeStamp();
    overhead += start - setup_start;

    index_type iter = 0;
    while(*m_N) {
        index_type n = *m_N;
        *m_N = 0;
        if (verbose) {
            printf("Iter %d\n",n);
        }

        if (out_type == OutType::QUEUE) {
            check_cuda(cudaMemset(vertex_claim,0,g.nnodes*sizeof(index_type)));
        }

        int len = (out_type == OutType::FILTER_IGNORE)?g.nnodes:n;
        int grid = (len + block_size-1)/block_size;
        wf_frontier_kernel<block_size, out_type><<<grid,block_size>>>(d_g, d_dists, list1, list2, len, m_N, vertex_claim, iter+1);
        check_cuda(cudaDeviceSynchronize());


        // filter
        if (out_type == OutType::FRONTIER) {
            n = *m_N;
            filter_frontier<<<(n + block_size-1)/block_size,block_size>>>(list2, list1,n,vertex_claim,iter+1);
            check_cuda(cudaDeviceSynchronize());
        } else if (out_type == OutType::FILTER_COMPACT) {
            cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,list2,list1,m_N,g.nnodes);
            check_cuda(cudaMemset(list2,0,g.nnodes*sizeof(index_type)));
            check_cuda(cudaDeviceSynchronize());

        } else if (out_type == OutType::FILTER_IGNORE || out_type == OutType::QUEUE) {
            index_type* tmp = list1;
            list1 = list2;
            list2 = tmp;
        }
        iter++;
    }

    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(list1));
    check_cuda(cudaFree(list2));
    check_cuda(cudaFree(m_N));
    if (vertex_claim) {
        check_cuda(cudaFree(vertex_claim));
    }

    overhead += getTimeStamp() - end;

    return {gpu_time, overhead};
}



/////////////////////////////////////////////////////////

void initialize_dists(edge_data_type* d_d, index_type n, index_type source) {
    check_cuda(cudaMemset(&d_d[0], 0xFF,  n * sizeof(edge_data_type)));
    check_cuda(cudaMemset(&d_d[source], 0,  sizeof(edge_data_type)));
}

void workfront_sweep(CSRGraph& g, edge_data_type* dists, index_type source) {
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));

    // Initialize for source node
    initialize_dists(d_d, g.nnodes, source);

    wf_sweep_frontier<256,OutType::FILTER_COMPACT>(g, d_g, d_d, source,true);
    // wf_coop_filter<32,CoopType::FULL,OutType::FRONTIER>(g, d_g, d_d, source,true);
    // wf_sweep_frontier<32,OutType::TOUCHED>, "frontier_filt_32" },

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}


struct Test {
    TimeCost (*f)(CSRGraph&, CSRGraph&, edge_data_type*, index_type, bool);
    const char* name;
};

void workfront_sweep_evaluation(CSRGraph& g, edge_data_type* dists, index_type source, edge_data_type* cpu) {
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node

    Test tests[] = {
        { wf_sweep_atomicq<32>, "atomic_32" },
        { wf_sweep_atomicq<64>, "atomic_64" },
        { wf_sweep_atomicq<128>, "atomic_128" },
        { wf_sweep_atomicq<256>, "atomic_256" },
        { wf_sweep_atomicq<512>, "atomic_512" },
        { wf_sweep_filter<32>, "filter_32" },
        { wf_sweep_filter<64>, "filter_64" },
        { wf_sweep_filter<128>, "filter_128" },
        { wf_sweep_filter<256>, "filter_256" },
        { wf_sweep_filter<512>, "filter_512" },
        { wf_sweep_coop<32,CoopType::VANILLA>, "vanilla_coop_32" },
        { wf_sweep_coop<64,CoopType::VANILLA>, "vanilla_coop_64" },
        { wf_sweep_coop<128,CoopType::VANILLA>, "vanilla_coop_128" },
        { wf_sweep_coop<256,CoopType::VANILLA>, "vanilla_coop_256" },
        { wf_sweep_coop<512,CoopType::VANILLA>, "vanilla_coop_512" },
        { wf_sweep_coop<32,CoopType::FULL>, "coop_32" },
        { wf_sweep_coop<64,CoopType::FULL>, "coop_64" },
        { wf_sweep_coop<128,CoopType::FULL>, "coop_128" },
        { wf_sweep_coop<256,CoopType::FULL>, "coop_256" },
        { wf_sweep_coop<512,CoopType::FULL>, "coop_512" },
        { wf_coop_filter<32,CoopType::FULL,OutType::FILTER_COMPACT>, "coop_filter_cp_32" },
        { wf_coop_filter<64,CoopType::FULL,OutType::FILTER_COMPACT>, "coop_filter_cp_64" },
        { wf_coop_filter<128,CoopType::FULL,OutType::FILTER_COMPACT>, "coop_filter_cp_128" },
        { wf_coop_filter<256,CoopType::FULL,OutType::FILTER_COMPACT>, "coop_filter_cp_256" },
        { wf_coop_filter<512,CoopType::FULL,OutType::FILTER_COMPACT>, "coop_filter_cp_512" },
        { wf_coop_filter<32,CoopType::FULL,OutType::FILTER_IGNORE>, "coop_filter_ig_32" },
        { wf_coop_filter<64,CoopType::FULL,OutType::FILTER_IGNORE>, "coop_filter_ig_64" },
        { wf_coop_filter<128,CoopType::FULL,OutType::FILTER_IGNORE>, "coop_filter_ig_128" },
        { wf_coop_filter<256,CoopType::FULL,OutType::FILTER_IGNORE>, "coop_filter_ig_256" },
        { wf_coop_filter<512,CoopType::FULL,OutType::FILTER_IGNORE>, "coop_filter_ig_512" },
        { wf_coop_filter<32,CoopType::FULL,OutType::FRONTIER>, "coop_frontier_ig_32" },
        { wf_coop_filter<64,CoopType::FULL,OutType::FRONTIER>, "coop_frontier_ig_64" },
        { wf_coop_filter<128,CoopType::FULL,OutType::FRONTIER>, "coop_frontier_ig_128" },
        { wf_coop_filter<256,CoopType::FULL,OutType::FRONTIER>, "coop_frontier_ig_256" },
        { wf_coop_filter<512,CoopType::FULL,OutType::FRONTIER>, "coop_frontier_ig_512" },
        { wf_sweep_frontier<32,OutType::QUEUE>, "frontier_q_32" },
        { wf_sweep_frontier<64,OutType::QUEUE>, "frontier_q_64" },
        { wf_sweep_frontier<128,OutType::QUEUE>, "frontier_q_128" },
        { wf_sweep_frontier<256,OutType::QUEUE>, "frontier_q_256" },
        { wf_sweep_frontier<512,OutType::QUEUE>, "frontier_q_512" },
        { wf_sweep_frontier<32,OutType::FRONTIER>, "frontier_32" },
        { wf_sweep_frontier<64,OutType::FRONTIER>, "frontier_64" },
        { wf_sweep_frontier<128,OutType::FRONTIER>, "frontier_128" },
        { wf_sweep_frontier<256,OutType::FRONTIER>, "frontier_256" },
        { wf_sweep_frontier<512,OutType::FRONTIER>, "frontier_512" },
        { wf_sweep_frontier<32,OutType::FILTER_COMPACT>, "frontier_filt_cp_32" },
        { wf_sweep_frontier<64,OutType::FILTER_COMPACT>, "frontier_filt_cp_64" },
        { wf_sweep_frontier<128,OutType::FILTER_COMPACT>, "frontier_filt_cp_128" },
        { wf_sweep_frontier<256,OutType::FILTER_COMPACT>, "frontier_filt_cp_256" },
        { wf_sweep_frontier<512,OutType::FILTER_COMPACT>, "frontier_filt_cp_512" },
        { wf_sweep_frontier<32,OutType::FILTER_IGNORE>, "frontier_filt_ig_32" },
        { wf_sweep_frontier<64,OutType::FILTER_IGNORE>, "frontier_filt_ig_64" },
        { wf_sweep_frontier<128,OutType::FILTER_IGNORE>, "frontier_filt_ig_128" },
        { wf_sweep_frontier<256,OutType::FILTER_IGNORE>, "frontier_filt_ig_256" },
        { wf_sweep_frontier<512,OutType::FILTER_IGNORE>, "frontier_filt_ig_512" },
    };

    printf("\n");
    for (int i = 0; i < sizeof(tests)/sizeof(Test); i++) {
        double best_gpu_time = 1000.0, best_overhead = 1000.0;
        printf("%s: ",tests[i].name);
        for (int j = 0; j < 2; j++) {
            initialize_dists(d_d, g.nnodes, source);
            auto tc = tests[i].f(g, d_g, d_d, source,false);
            printf("{%f, %f}, ", tc.gpu_time, tc.overhead);
            if (tc.gpu_time < best_gpu_time) best_gpu_time = tc.gpu_time;
            if (tc.overhead < best_overhead) best_overhead = tc.overhead;
        }
        printf(", Best: {%f, %f}, ", best_gpu_time, best_overhead);
        cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
        compare(cpu,dists,g.nnodes);
    }
    printf("\n");


    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}
