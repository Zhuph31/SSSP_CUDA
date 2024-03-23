#include "nearfar.cuh"

edge_data_type calculate_delta(CSRGraph g) {
  // delta as warp size * average weight / average degree
  // calcuate average edge weight
  unsigned long ew = 0;
  for (int i = 0; i < g.nedges; i++) {
    ew += g.edge_data[i];
  }
  ew = ew / g.nedges;
  // calculate average edge degree as total # of edges / total # of nodes
  edge_data_type d = (g.nedges / g.nnodes);
  printf("average degree %d, average weight %lu, delta %lu \n", d, ew,
         32 * ew / d);
  return 32 * ew / d;
}

__global__ void nf_iter(CSRGraph g, edge_data_type *d, edge_data_type delta,
                        index_type *last_near_pile, index_type last_near_len,
                        index_type *new_near_pile, index_type *new_near_len,
                        index_type *new_far_pile, index_type *new_far_len,
                        index_type *scratch) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

  // only work on near pile
  if (index < last_near_len) {
    index_type s_idx = last_near_pile[index];
    // somewhat baed on
    // https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b
    for (int j = g.row_start[s_idx]; j < g.row_start[s_idx + 1]; j++) {
      edge_data_type w = d[s_idx];
      edge_data_type ew = g.edge_data[j];
      index_type n = g.edge_dst[j];
      edge_data_type nw = d[n];
      edge_data_type new_w = ew + w;
      // Check if the distance is already set to max then just take the max
      // since,
      if (w >= MAX_VAL) {
        new_w = MAX_VAL;
      }

      if (new_w < nw) {
        atomicMin(&d[n], new_w);
        // seperating into near and far pile
        index_type q_idx;
        if (new_w < delta) {
          // near pile, this value has been updated. Flag this value need to be
          // thrown out in far pile
          atomicAdd(&scratch[n], 1);
          q_idx = atomicAdd(new_near_len, 1);
          new_near_pile[q_idx] = n;
        } else {
          q_idx = atomicAdd(new_far_len, 1);
          new_far_pile[q_idx] = n;
        }
        // printf("Adding %d to q[%d]\n",n,q_idx);
      }
    }
  }
}

__global__ void far_split(edge_data_type *d, edge_data_type delta,
                          index_type *last_far_pile, index_type last_far_len,
                          index_type *new_near_pile, index_type *new_near_len,
                          index_type *new_far_pile, index_type *new_far_len,
                          index_type *scratch) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
  // only split far pile into near and far
  if (index < last_far_len) {
    index_type far_idx = last_far_pile[index];
    edge_data_type nw = d[far_idx];
    index_type q_idx;

    // remove duplicates in the far pile, and remove flagged invalid edge from
    // near pile update
    if (atomicCAS(&scratch[far_idx], 0, 1) == 0) {
      if (nw <= delta) {
        q_idx = atomicAdd(new_near_len, 1);
        new_near_pile[q_idx] = far_idx;
      } else {
        q_idx = atomicAdd(new_far_len, 1);
        new_far_pile[q_idx] = far_idx;
      }
    }
  }
}

void nearfar(CSRGraph &g, edge_data_type *dists) {

  double start, end = 0;
  CSRGraph d_g;
  g.copy_to_gpu(d_g);
  edge_data_type *d_d = NULL;
  check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
  // Initialize for source node = 0. Otherwise need to change this
  check_cuda(
      cudaMemset(&d_d[1], 0xFF, (g.nnodes - 1) * sizeof(edge_data_type)));

  index_type *near1 = NULL, *near2 = NULL, *far1 = NULL, *far2 = NULL;
  index_type *qscratch = NULL;
  check_cuda(cudaMalloc(&near1, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&near2, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&far1, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&far2, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&qscratch, g.nnodes * sizeof(index_type)));

  // Set first q entry to 0 (source) TODO: other sources
  check_cuda(cudaMemset(near1, 0, sizeof(index_type)));
  check_cuda(cudaMemset(qscratch, 0, g.nnodes * sizeof(index_type)));
  index_type *near_len = NULL, *far_len = NULL;
  check_cuda(cudaMallocManaged(&near_len, sizeof(index_type)));
  check_cuda(cudaMallocManaged(&far_len, sizeof(index_type)));
  *near_len = 1, *far_len = 1;

  printf("weight %d \n", g.getWeight(0, 0));
  start = getTimeStamp();
  int iter = 0;

  // calculate delta for graph, calculated delta seems to be too large, divide
  // by 10 gets good results on rmat22
  edge_data_type delta = calculate_delta(g) / 10;
  printf("Actual delta used %d \n", delta);

  while (*far_len > 0 || *near_len > 0) {
    printf("Iter %d, near batch %d, far batch %d, delta %d\n", iter, *near_len,
           *far_len, delta);
    index_type old_near_len = *near_len;
    *near_len = 0;
    // keep
    nf_iter<<<(old_near_len + 512 - 1) / 512, 512>>>(
        d_g, d_d, delta, near1, old_near_len, near2, near_len, far2, far_len,
        qscratch);
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
            d_d, delta, far2, old_far_len, near1, near_len, far1, far_len,
            qscratch);
        cudaDeviceSynchronize();
      }
      index_type *tmp = far1;
      far1 = far2;
      far2 = tmp;

      // reset scratch array to 0
      check_cuda(cudaMemset(qscratch, 0, g.nnodes * sizeof(index_type)));

      // printf("after delta_update, near batch %d, far batch %d , delta %d\n",
      // *near_len, *far_len, delta);
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
  printf("GPU time: %f\n", gpu_time);

  cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type),
             cudaMemcpyDeviceToHost);
}