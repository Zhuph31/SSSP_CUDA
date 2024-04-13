#include "utils.cuh"
#include "workfront-sweep.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/device/device_select.cuh>

enum OutType {
    QUEUE,
    FILTER_COMPACT,
    FILTER_IGNORE,
    FRONTIER
};

/////////////// ATOMIC QUEUE //////////////////

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
    index_type* worklist1, *worklist2 = NULL;
    index_type* vertex_claim = NULL;
    index_type* worklist_len = NULL;

    double setup_start = getTimeStamp();
    check_cuda(cudaMalloc(&worklist1, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&worklist2, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&vertex_claim, g.nnodes* sizeof(index_type)));
    check_cuda(cudaMallocManaged(&worklist_len, sizeof(index_type)));

    // Set first worklist entry to 0 (source)
    check_cuda(cudaMemcpy(worklist1, &source, sizeof(index_type), cudaMemcpyHostToDevice));
    *worklist_len = 1;


    start = getTimeStamp();
    overhead += start - setup_start;

    int itr = 0;
    while (*worklist_len) {
        if (verbose) {
            printf("Iter %d, worklist length %d\n",itr, *worklist_len);
        }
        index_type len = *worklist_len;
        *worklist_len = 0;
 
        wf_iter_simple<OutType::QUEUE><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_dists, worklist1, len,worklist2, worklist_len, vertex_claim);
        check_cuda(cudaMemset(vertex_claim,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = worklist1;
        worklist1 = worklist2;
        worklist2 = tmp;


        itr += 1;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(worklist1));
    check_cuda(cudaFree(worklist2));
    check_cuda(cudaFree(vertex_claim));
    check_cuda(cudaFree(worklist_len));
    overhead += getTimeStamp() - end;
    return {gpu_time, overhead};
}


////////// COMPACTION FILTERING //////////////////////

__global__ void setup_id(index_type* out, index_type n) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
    if (index < n)  out[index] = index;
}


template <int block_size>
TimeCost wf_sweep_filter(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* worklist = NULL;
    index_type* scan_indices = NULL;
    index_type* touched = NULL;
    double setup_start = getTimeStamp();
    check_cuda(cudaMalloc(&worklist, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&touched, g.nnodes* sizeof(index_type)));
    check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
    setup_id<<<(g.nnodes + block_size - 1),block_size>>>(scan_indices,g.nnodes);

    // Set first worklist entry to source
    check_cuda(cudaMemcpy(worklist, &source, sizeof(index_type),cudaMemcpyHostToDevice));
    index_type* worklist_len = NULL;
    check_cuda(cudaMallocManaged(&worklist_len, sizeof(index_type)));
    *worklist_len = 1;

    void* flg_tmp_store = NULL;
    size_t flg_store_size = 0;
    cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,worklist,worklist_len,g.nnodes);
    check_cuda(cudaMalloc(&flg_tmp_store,flg_store_size));
    check_cuda(cudaMemset(touched,0,g.nnodes*sizeof(index_type)));


    start = getTimeStamp();
    overhead += start - setup_start;
    while (*worklist_len) {
        if (verbose) {
            printf("Iter %d\n",*worklist_len);
        }
        index_type len = *worklist_len;
        *worklist_len = 0;
 
        wf_iter_simple<OutType::FILTER_COMPACT><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_dists, worklist, len, touched,NULL, NULL);

        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,touched,worklist,worklist_len,g.nnodes);
        check_cuda(cudaMemset(touched,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(worklist));
    check_cuda(cudaFree(touched));
    check_cuda(cudaFree(scan_indices));
    check_cuda(cudaFree(worklist_len));
    overhead += getTimeStamp() - end;
    return {gpu_time, overhead};
}


///////////////// SIMPLE COOP ///////////////////////////////////////////////////////

template <int block_size, OutType out_type>
__global__ void wf_simple_coop_iter(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type input_len, index_type* output, index_type* output_len,  index_type* vertex_claim) {

    __shared__ index_type block_row_start;
    __shared__ index_type block_row_end; 
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
        // The threads in this block each take one edge
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


template <int block_size>
TimeCost wf_sweep_simple_coop(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_d, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* worklist1, *worklist2 = NULL;
    index_type* vertex_claim = NULL;
    index_type* worklist_len = NULL;

    double setup_start = getTimeStamp();
    check_cuda(cudaMalloc(&worklist1, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&worklist2, g.nnodes * sizeof(index_type)));
    check_cuda(cudaMalloc(&vertex_claim, g.nnodes* sizeof(index_type)));
    check_cuda(cudaMallocManaged(&worklist_len, sizeof(index_type)));

    // Set first worklist entry to source
    check_cuda(cudaMemcpy(worklist1, &source, sizeof(index_type), cudaMemcpyHostToDevice));
    *worklist_len = 1;


    start = getTimeStamp();
    overhead += start - setup_start;

    int itr = 0;
    while (*worklist_len) {
        if (verbose) {
            printf("Iter %d, qlen %d\n",itr, *worklist_len);
        }
        index_type len = *worklist_len;
        *worklist_len = 0;

        wf_simple_coop_iter<block_size, OutType::QUEUE><<<(len + block_size - 1) / block_size,block_size>>>(d_g, d_d, worklist1, len, worklist2, worklist_len, vertex_claim);

        check_cuda(cudaMemset(vertex_claim,0,g.nnodes*sizeof(index_type)));
        cudaDeviceSynchronize();

        index_type* tmp = worklist1;
        worklist1 = worklist2;
        worklist2 = tmp;


        itr += 1;
    }
    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(worklist1));
    check_cuda(cudaFree(worklist2));
    check_cuda(cudaFree(vertex_claim));
    check_cuda(cudaFree(worklist_len));

    overhead += getTimeStamp() - end;

    return {gpu_time, overhead};
}






//////////////////////// FULL COOPERATIVE ////////////////////////

template <int block_size, OutType out_type>
__global__ void wf_full_coop_iter(CSRGraph g, edge_data_type* best_costs, index_type* worklist_in, index_type* output, index_type input_len, index_type* output_len, index_type* vertex_claim, index_type iter) {
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


    index_type block_aggregate = 0;
    BlockScan(temp_storage).ExclusiveSum(degree, degree, block_aggregate);
    output_offset[tidx] = degree;

    if (block_aggregate == 0) {
        return;
    }

    if (out_type == OutType::FRONTIER || out_type == OutType::FILTER_IGNORE) {
        if (tidx == 0 && block_aggregate) {
            block_offset[0] = atomicAdd(output_len,block_aggregate);
        }
    }

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
        index_type dest_vtx = g.edge_dst[first_edge_offset[v_id]+ edge_offset];
        edge_data_type edge_weight = g.edge_data[first_edge_offset[v_id]+ edge_offset];
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


template <int block_size, OutType out_type>
TimeCost wf_sweep_full_coop(CSRGraph& g, CSRGraph& d_g, edge_data_type* d_dists, index_type source, bool verbose=false) {
    double start, end = 0, overhead = 0;
    index_type* worklist1, *worklist2, *vertex_claim = NULL;
    index_type* managed_length = NULL;

    double setup_start = getTimeStamp();
    if (out_type == OutType::FRONTIER) {
        check_cuda(cudaMalloc(&worklist1, g.nedges * sizeof(index_type)));
        check_cuda(cudaMalloc(&worklist2, g.nedges * sizeof(index_type)));
    } else {
        check_cuda(cudaMalloc(&worklist1, g.nnodes * sizeof(index_type)));
        check_cuda(cudaMalloc(&worklist2, g.nnodes * sizeof(index_type)));
    }
    check_cuda(cudaMallocManaged(&managed_length, sizeof(index_type)));

    index_type* scan_indices = NULL;
    void* flg_tmp_store = NULL;
    size_t flg_store_size = 0;
    if (out_type == OutType::FILTER_COMPACT) {
        check_cuda(cudaMalloc(&scan_indices, g.nnodes* sizeof(index_type)));
        setup_id<<<(g.nnodes + block_size - 1),block_size>>>(scan_indices,g.nnodes);
        cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,worklist1,worklist2,managed_length,g.nnodes);
        check_cuda(cudaMalloc(&flg_tmp_store,flg_store_size));
        check_cuda(cudaMemset(worklist2,0,g.nnodes*sizeof(index_type)));
    }

    if (out_type == OutType::FILTER_IGNORE) {
        cudaMemset(&worklist1[0], 0xFF, g.nnodes*sizeof(index_type));
        cudaMemset(&worklist2[0], 0xFF, g.nnodes*sizeof(index_type));
    }

    if (out_type == OutType::QUEUE || out_type == OutType::FRONTIER) {
        cudaMalloc(&vertex_claim, g.nnodes * sizeof(index_type));
        check_cuda(cudaMemset(vertex_claim,0,g.nnodes*sizeof(index_type)));
    }

    // Set the source vertex
    if (out_type == OutType::FILTER_IGNORE) {
        check_cuda(cudaMemset(&worklist1[source],0, sizeof(index_type)));
    } else {
        check_cuda(cudaMemcpy(&worklist1[0],&source, sizeof(index_type), cudaMemcpyHostToDevice));
    }
    *managed_length = 1;

    start = getTimeStamp();
    overhead += start - setup_start;

    index_type iter = 0;
    while(*managed_length) {
        index_type n = *managed_length;
        *managed_length = 0;
        if (verbose) {
            printf("Iter %d\n",n);
        }

        if (out_type == OutType::QUEUE) {
            check_cuda(cudaMemset(vertex_claim,0,g.nnodes*sizeof(index_type)));
        }

        int len = (out_type == OutType::FILTER_IGNORE)?g.nnodes:n;
        int grid = (len + block_size-1)/block_size;
        wf_full_coop_iter<block_size, out_type><<<grid,block_size>>>(d_g, d_dists, worklist1, worklist2, len, managed_length, vertex_claim, iter+1);
        check_cuda(cudaDeviceSynchronize());


        // filter
        if (out_type == OutType::FRONTIER) {
            n = *managed_length;
            filter_frontier<<<(n + block_size-1)/block_size,block_size>>>(worklist2, worklist1,n,vertex_claim,iter+1);
            check_cuda(cudaDeviceSynchronize());
        } else if (out_type == OutType::FILTER_COMPACT) {
            cub::DeviceSelect::Flagged(flg_tmp_store,flg_store_size,scan_indices,worklist2,worklist1,managed_length,g.nnodes);
            check_cuda(cudaMemset(worklist2,0,g.nnodes*sizeof(index_type)));
            check_cuda(cudaDeviceSynchronize());

        } else if (out_type == OutType::FILTER_IGNORE || out_type == OutType::QUEUE) {
            index_type* tmp = worklist1;
            worklist1 = worklist2;
            worklist2 = tmp;
        }
        iter++;
    }

    end = getTimeStamp();
    double gpu_time = end - start;
    if (verbose) printf("GPU time: %f\n",gpu_time);

    check_cuda(cudaFree(worklist1));
    check_cuda(cudaFree(worklist2));
    check_cuda(cudaFree(managed_length));
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
    double tf_start = getTimeStamp();
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d = NULL;
    check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
    double tf_time = getTimeStamp() - tf_start;

    // Initialize for source node
    initialize_dists(d_d, g.nnodes, source);

    wf_sweep_full_coop<256,OutType::FILTER_COMPACT>(g, d_g, d_d, source,true);

    tf_start = getTimeStamp();
    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
    tf_time += getTimeStamp() - tf_start;
    printf("Transfer time: %f\n",tf_time);
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
        { wf_sweep_atomicq<32>, "atomic_queue_32" },
        { wf_sweep_atomicq<64>, "atomic_queue_64" },
        { wf_sweep_atomicq<128>, "atomic_queue_128" },
        { wf_sweep_atomicq<256>, "atomic_queue_256" },
        { wf_sweep_atomicq<512>, "atomic_queue_512" },
        { wf_sweep_filter<32>, "filter_32" },
        { wf_sweep_filter<64>, "filter_64" },
        { wf_sweep_filter<128>, "filter_128" },
        { wf_sweep_filter<256>, "filter_256" },
        { wf_sweep_filter<512>, "filter_512" },
        { wf_sweep_simple_coop<32>, "simple_coop_32" },
        { wf_sweep_simple_coop<64>, "simple_coop_64" },
        { wf_sweep_simple_coop<128>, "simple_coop_128" },
        { wf_sweep_simple_coop<256>, "simple_coop_256" },
        { wf_sweep_simple_coop<512>, "simple_coop_512" },
        { wf_sweep_full_coop<32,OutType::QUEUE>, "full_coop_q_32" },
        { wf_sweep_full_coop<64,OutType::QUEUE>, "full_coop_q_64" },
        { wf_sweep_full_coop<128,OutType::QUEUE>, "full_coop_q_128" },
        { wf_sweep_full_coop<256,OutType::QUEUE>, "full_coop_q_256" },
        { wf_sweep_full_coop<512,OutType::QUEUE>, "full_coop_q_512" },
        { wf_sweep_full_coop<32,OutType::FRONTIER>, "full_coop_frontier_32" },
        { wf_sweep_full_coop<64,OutType::FRONTIER>, "full_coop_frontier_64" },
        { wf_sweep_full_coop<128,OutType::FRONTIER>, "full_coop_frontier_128" },
        { wf_sweep_full_coop<256,OutType::FRONTIER>, "full_coop_frontier_256" },
        { wf_sweep_full_coop<512,OutType::FRONTIER>, "full_coop_frontier_512" },
        { wf_sweep_full_coop<32,OutType::FILTER_COMPACT>, "full_coop_filt_cp_32" },
        { wf_sweep_full_coop<64,OutType::FILTER_COMPACT>, "full_coop_filt_cp_64" },
        { wf_sweep_full_coop<128,OutType::FILTER_COMPACT>, "full_coop_filt_cp_128" },
        { wf_sweep_full_coop<256,OutType::FILTER_COMPACT>, "full_coop_filt_cp_256" },
        { wf_sweep_full_coop<512,OutType::FILTER_COMPACT>, "full_coop_filt_cp_512" },
        { wf_sweep_full_coop<32,OutType::FILTER_IGNORE>, "full_coop_filt_ig_32" },
        { wf_sweep_full_coop<64,OutType::FILTER_IGNORE>, "full_coop_filt_ig_64" },
        { wf_sweep_full_coop<128,OutType::FILTER_IGNORE>, "full_coop_filt_ig_128" },
        { wf_sweep_full_coop<256,OutType::FILTER_IGNORE>, "full_coop_filt_ig_256" },
        { wf_sweep_full_coop<512,OutType::FILTER_IGNORE>, "full_coop_filt_ig_512" },
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
