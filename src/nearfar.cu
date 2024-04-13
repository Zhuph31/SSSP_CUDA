#include "nearfar.cuh"

__global__ void nf_iter(CSRGraph g, edge_data_type *best_cost,
                        edge_data_type delta, index_type *last_near_pile,
                        index_type last_near_len, index_type *new_near_pile,
                        index_type *new_near_len, index_type *new_far_pile,
                        index_type *new_far_len, index_type *scratch) {
  index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);

  // only work on near pile
  if (gidx < last_near_len) {
    index_type s_idx = last_near_pile[gidx];
    for (int j = g.row_start[s_idx]; j < g.row_start[s_idx + 1]; j++) {
      edge_data_type source_cost = best_cost[s_idx];
      edge_data_type edge_weight = g.edge_data[j];
      index_type dest_vtx = g.edge_dst[j];
      edge_data_type old_dest_cost = best_cost[dest_vtx];
      edge_data_type new_dest_cost = edge_weight + source_cost;
      // Check if the distance is already set to max then just take the max
      // since,
      if (source_cost >= MAX_VAL) {
        new_dest_cost = MAX_VAL;
      }

      if (new_dest_cost < old_dest_cost) {
        atomicMin(&best_cost[dest_vtx], new_dest_cost);
        // seperating into near and far pile
        index_type q_idx;
        if (new_dest_cost < delta) {
          // near pile, this value has been updated. Flag this value need to be
          // thrown out in far pile
          atomicAdd(&scratch[dest_vtx], 1);
          q_idx = atomicAdd(new_near_len, 1);
          new_near_pile[q_idx] = dest_vtx;
        } else {
          q_idx = atomicAdd(new_far_len, 1);
          new_far_pile[q_idx] = dest_vtx;
        }
      }
    }
  }
}

__global__ void far_split(edge_data_type *best_cost, edge_data_type delta,
                          index_type *last_far_pile, index_type last_far_len,
                          index_type *new_near_pile, index_type *new_near_len,
                          index_type *new_far_pile, index_type *new_far_len,
                          index_type *scratch) {
  index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);
  // only split far pile into near and far
  if (gidx < last_far_len) {
    index_type far_idx = last_far_pile[gidx];
    edge_data_type old_dest_cost = best_cost[far_idx];
    index_type q_idx;

    // remove duplicates in the far pile, and remove flagged invalid edge from
    // near pile update
    if (atomicCAS(&scratch[far_idx], 0, 1) == 0) {
      if (old_dest_cost <= delta) {
        q_idx = atomicAdd(new_near_len, 1);
        new_near_pile[q_idx] = far_idx;
      } else {
        q_idx = atomicAdd(new_far_len, 1);
        new_far_pile[q_idx] = far_idx;
      }
    }
  }
}

TimeCost nearfar(CSRGraph &g, edge_data_type *dists) {
  double start, end = 0, overhead = 0;
  CSRGraph d_g;
  g.copy_to_gpu(d_g);
  edge_data_type *d_dists = NULL;
  check_cuda(cudaMalloc(&d_dists, g.nnodes * sizeof(edge_data_type)));
  // Initialize for source node = 0. Otherwise need to change this
  check_cuda(
      cudaMemset(&d_dists[1], 0xFF, (g.nnodes - 1) * sizeof(edge_data_type)));

  double setup_start = getTimeStamp();
  index_type *near1 = NULL, *near2 = NULL, *far1 = NULL, *far2 = NULL,
             *vertex_claim = NULL;
  check_cuda(cudaMalloc(&near1, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&near2, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&far1, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&far2, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&vertex_claim, g.nnodes * sizeof(index_type)));

  // Set first q entry to 0 (source) TODO: other sources
  check_cuda(cudaMemset(near1, 0, sizeof(index_type)));
  check_cuda(cudaMemset(vertex_claim, 0, g.nnodes * sizeof(index_type)));
  index_type *near_len = NULL, *far_len = NULL;
  check_cuda(cudaMallocManaged(&near_len, sizeof(index_type)));
  check_cuda(cudaMallocManaged(&far_len, sizeof(index_type)));
  *near_len = 1, *far_len = 1;

  start = getTimeStamp();
  overhead += start - setup_start;
  int iter = 0;

  // calculate delta for graph, calculated delta seems to be too large, divide
  // by 10 gets good results on rmat22
  edge_data_type delta = calculate_delta(g) / 10;

  while (*far_len > 0 || *near_len > 0) {
    index_type old_near_len = *near_len;
    *near_len = 0;
    // keep
    nf_iter<<<(old_near_len + 512 - 1) / 512, 512>>>(
        d_g, d_dists, delta, near1, old_near_len, near2, near_len, far2,
        far_len, vertex_claim);
    cudaDeviceSynchronize();

    if (*far_len == 0 && *near_len == 0)
      break;

    if (*near_len == 0) {

      index_type old_far_len = *far_len;

      while (*near_len == 0) {
        // keep adding delta until we have something in near batch
        delta += delta;
        *far_len = 0;
        *near_len = 0;
        far_split<<<(old_far_len + 512 - 1) / 512, 512>>>(
            d_dists, delta, far2, old_far_len, near1, near_len, far1, far_len,
            vertex_claim);
        cudaDeviceSynchronize();
      }
      index_type *tmp = far1;
      far1 = far2;
      far2 = tmp;

      // reset scratch array to 0
      check_cuda(cudaMemset(vertex_claim, 0, g.nnodes * sizeof(index_type)));
    } else {
      // continue working on near pile, switch near pile, keep adding to same
      // far pile
      index_type *tmp = near1;
      near1 = near2;
      near2 = tmp;
    }

    iter += 1;
  }
  end = getTimeStamp();
  double gpu_time = end - start;

  check_cuda(cudaFree(near1));
  check_cuda(cudaFree(near2));
  check_cuda(cudaFree(far1));
  check_cuda(cudaFree(far2));
  check_cuda(cudaFree(vertex_claim));
  check_cuda(cudaFree(near_len));
  check_cuda(cudaFree(far_len));
  overhead += getTimeStamp() - end;

  cudaMemcpy(dists, d_dists, g.nnodes * sizeof(edge_data_type),
             cudaMemcpyDeviceToHost);

  return {gpu_time, overhead};
}