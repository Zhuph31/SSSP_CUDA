#include "csr.h"
#include "nearfar.h"
#include <sys/time.h>
#include <thrust/sort.h>

const uint32_t nBuckets = 10;

__global__ void bucketing_iter(CSRGraph g, edge_data_type *d,
                               index_type *bucket, index_type bucket_len,
                               index_type *output, index_type *output_len) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

  // only work on closest bucket pile
  if (index < bucket_len) {
    index_type s_idx = bucket[index];
    for (int j = g.row_start[s_idx]; j < g.row_start[s_idx + 1];
         j++) {                           // range each edge for current node
      edge_data_type w = d[s_idx];        // current approximation for departure
                                          // node for current edge
      edge_data_type ew = g.edge_data[j]; // the weight of current edge
      index_type n = g.edge_dst[j];  // the destination node for current edge
      edge_data_type nw = d[n];      // old approximation for node n
      edge_data_type new_w = ew + w; // new approximation starting from j
      // Check if the distance is already set to max then just take the max
      // since,
      if (w >= MAX_VAL) {
        new_w = MAX_VAL;
      }

      if (new_w < nw) {
        atomicMin(&d[n], new_w);

        // append into output
        index_type q_idx = atomicAdd(output_len, 1);
        output[q_idx] = n;
        // printf("Adding %d to q[%d]\n",n,q_idx);
      }
    }
  }
}

void bucketing_impl(CSRGraph &g, edge_data_type *dists) {
  double start, end = 0;
  CSRGraph d_g;
  g.copy_to_gpu(d_g);
  edge_data_type *d_d = NULL;
  check_cuda(cudaMalloc(&d_d, g.nnodes * sizeof(edge_data_type)));
  // Initialize for source node = 0. Otherwise need to change this
  check_cuda(
      cudaMemset(&d_d[1], 0xFF, (g.nnodes - 1) * sizeof(edge_data_type)));

  index_type **buckets = nullptr, *far = nullptr, *temp_output = nullptr;

  check_cuda(cudaMalloc(&buckets, nBuckets * sizeof(index_type)));
  for (int i = 0; i < nBuckets; ++i) {
    check_cuda(cudaMalloc((void **)&(buckets[i]),
                          (g.nedges + 1) / 2 * sizeof(index_type)));
  }
  check_cuda(cudaMalloc(&far, (g.nedges + 1) * sizeof(index_type)));
  check_cuda(cudaMalloc(&temp_output, (g.nedges + 1) * sizeof(index_type)));
  // Set first q entry to 0 (source) TODO: other sources
  check_cuda(cudaMemset(buckets[0], 0, sizeof(index_type)));

  index_type *buckets_len = nullptr, *far_len = NULL,
             *temp_output_len = nullptr;
  check_cuda(cudaMallocManaged(&buckets_len, nBuckets * sizeof(index_type)));
  check_cuda(cudaMallocManaged(&far_len, sizeof(index_type)));
  check_cuda(cudaMallocManaged(&temp_output_len, sizeof(index_type)));
  buckets_len[0] = 1, *far_len = 1, temp_output_len = 0;

  printf("weight %d \n", g.getWeight(0, 0));
  start = getTimeStamp();

  int iter = 0, bucket_idx = 0;

  // calculate delta for graph
  edge_data_type delta = calculate_delta(g);

  // while (*far_len > 0 || buckets_len[bucket_idx] > 0) {
  while (1) {
    printf("Iter %d, far batch %d, delta %d\n", iter, *far_len, delta);

    for (int i = 0; i < nBuckets; ++i) {
      if (buckets_len[i] == 0) {
        break;
      }

      index_type current_bucket_len = buckets_len[bucket_idx];
      buckets_len[bucket_idx] = 0;

      bucketing_iter<<<(current_bucket_len + 512 - 1) / 512, 512>>>(
          d_g, d_d, buckets[bucket_idx], current_bucket_len, temp_output,
          temp_output_len);
      cudaDeviceSynchronize();

      // split output into future buckets and far pile
      // update future buckets len
      // go to next bucket
    }

    // printf("after update, near batch %d, far batch %d\n", *near_len,
    // *far_len);

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
            d_d, delta, far2, old_far_len, near1, near_len, far1, far_len);
        cudaDeviceSynchronize();
      }
      index_type *tmp = far1;
      far1 = far2;
      far2 = tmp;
      // printf("after delta_update, near batch %d, far batch %d , delta %d\n",
      // *near_len, *far_len, delta);
    } else {
      // continue working on near pile, switch near pile, keep adding to same
      // far pile
      index_type *tmp = near1;
      near1 = near2;
      near2 = tmp;
    }

    ++iter;
    ++bucket_idx;
  }
  end = getTimeStamp();
  double gpu_time = end - start;
  printf("GPU time: %f\n", gpu_time);

  cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type),
             cudaMemcpyDeviceToHost);
}
