#include "bucketing.cuh"
#include <stdint.h>
#include <utility>

const uint32_t nBuckets = 10;

__global__ inline void
far_split(edge_data_type *best_cost, edge_data_type delta,
          index_type *last_far_pile, index_type *last_far_len,
          index_type *new_near_pile, index_type *new_near_len,
          index_type *new_far_pile, index_type *new_far_len) {
  index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);
  // only split far pile into near and far
  if (gidx < *last_far_len) {
    index_type far_idx = last_far_pile[gidx];
    edge_data_type nw = best_cost[far_idx];
    index_type q_idx;

    if (nw < delta) {
      q_idx = atomicAdd(new_near_len, 1);
      new_near_pile[q_idx] = far_idx;
    } else {
      q_idx = atomicAdd(new_far_len, 1);
      new_far_pile[q_idx] = far_idx;
    }
  }
}

__global__ void bucketing_iter(CSRGraph g, edge_data_type *best_cost,
                               edge_data_type delta, index_type *bucket,
                               index_type bucket_len, index_type *near,
                               index_type *near_len, index_type *far,
                               index_type *far_len) {
  index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);

  // only work on closest bucket pile
  if (gidx < bucket_len) {
    index_type s_idx = bucket[gidx];
    for (int j = g.row_start[s_idx]; j < g.row_start[s_idx + 1];
         j++) {                            // range each edge for current node
      edge_data_type w = best_cost[s_idx]; // current approximation for
                                           // departure node for current edge
      edge_data_type ew = g.edge_data[j];  // the weight of current edge
      index_type n = g.edge_dst[j];     // the destination node for current edge
      edge_data_type nw = best_cost[n]; // old approximation for node n
      edge_data_type new_w = ew + w;    // new approximation starting from j
      // Check if the distance is already set to max then just take the max
      // since,
      if (w >= MAX_VAL) {
        new_w = MAX_VAL;
      }

      if (new_w < nw) {
        atomicMin(&best_cost[n], new_w);

        // seperating into near and far pile
        index_type q_idx;
        if (new_w < delta) {
          q_idx = atomicAdd(near_len, 1);
          near[q_idx] = n;
        } else {
          q_idx = atomicAdd(far_len, 1);
          far[q_idx] = n;
        }
      }
    }
  }
}

// split near pile into buckets
__global__ void buckets_split(edge_data_type delta, edge_data_type *d_dists,
                              index_type *near, index_type *near_len,
                              index_type *buckets, index_type *buckets_len,
                              index_type each_bucket_cap) {
  index_type gidx = threadIdx.x + (blockDim.x * blockIdx.x);
  if (gidx < *near_len) {
    index_type n_idx = near[gidx];
    edge_data_type w = d_dists[n_idx];

    edge_data_type base_delta = delta / 10;
    edge_data_type bucket_idx = w / base_delta;
    if (bucket_idx > nBuckets - 1) {
      bucket_idx = nBuckets - 1;
    }

    index_type q_idx = atomicAdd(&buckets_len[bucket_idx], 1);
    index_type *bucket_start =
        buckets +
        bucket_idx * each_bucket_cap; // start of the corresponding bucket

    *(bucket_start + q_idx) = n_idx;
  }
}

TimeCost bucketing(CSRGraph &g, edge_data_type *dists) {
  double start, end = 0, overhead = 0;
  CSRGraph d_g;
  g.copy_to_gpu(d_g);
  edge_data_type *d_dists = NULL;
  check_cuda(cudaMalloc(&d_dists, g.nnodes * sizeof(edge_data_type)));
  // Initialize for source node = 0. Otherwise need to change this
  check_cuda(
      cudaMemset(&d_dists[1], 0xFF, (g.nnodes - 1) * sizeof(edge_data_type)));

  index_type *buckets = nullptr, *near = nullptr, *far = nullptr,
             *far2 = nullptr;

  // buckets : all the buckets within range of delta
  // near : store the vertices that falls in range of delta after each
  // iteration, to be merged with buckets
  // sorting
  // far : all vertices out of range of delta
  // far2 : backup vector used for far split

  index_type each_bucket_cap = g.nedges;
  index_type bucket_total_cap = each_bucket_cap * nBuckets;

  double setup_start = getTimeStamp();
  check_cuda(cudaMalloc(&buckets, bucket_total_cap * sizeof(index_type)));
  check_cuda(cudaMalloc(&near, g.nedges * sizeof(index_type)));
  check_cuda(cudaMalloc(&far, g.nedges * 2 * sizeof(index_type)));
  check_cuda(cudaMalloc(&far2, g.nedges * 2 * sizeof(index_type)));

  // Set first q entry to 0 (source) TODO: other sources
  check_cuda(cudaMemset(buckets, 0, sizeof(index_type)));

  index_type *buckets_len = nullptr, *near_len = nullptr, *far_len = NULL,
             *far2_len = nullptr;
  check_cuda(cudaMallocManaged(&buckets_len, nBuckets * sizeof(index_type)));
  check_cuda(cudaMallocManaged(&near_len, sizeof(index_type)));
  check_cuda(cudaMallocManaged(&far_len, sizeof(index_type)));
  check_cuda(cudaMallocManaged(&far2_len, sizeof(index_type)));
  buckets_len[0] = 1, *far_len = 0, *near_len = 0, *far2_len = 0;
  for (int i = 1; i < nBuckets; ++i) { // set later buckets len to 0
    buckets_len[i] = 0;
  }

  start = getTimeStamp();
  overhead += start - setup_start;

  // calculate delta for graph
  edge_data_type delta = calculate_delta(g);
  if (delta < 1) {
    delta = 1;
  }

  int iter = 0, bucket_idx = 0;
  bool found_non_empty_bucket = true;
  while (1) { // break on near pile processed and far pile is empty
    // keep processing buckets until all buckets are emtpy
    while (found_non_empty_bucket) {
      index_type current_bucket_len = buckets_len[bucket_idx];

      index_type *current_bucket =
          buckets +
          bucket_idx * each_bucket_cap; // calculate the start of current bucket

      bucketing_iter<<<(current_bucket_len + 512 - 1) / 512, 512>>>(
          d_g, d_dists, delta, current_bucket, current_bucket_len, near,
          near_len, far, far_len);
      cudaDeviceSynchronize();

      // set current bucket size to 0 as they are all processed
      buckets_len[bucket_idx] = 0;

      // split near pile into buckets only when near pile is not empty
      if (*near_len > 0) {
        index_type griddim = (*near_len + 512 - 1) / 512;
        buckets_split<<<griddim, 512>>>(delta, d_dists, near, near_len, buckets,
                                        buckets_len, each_bucket_cap);
        cudaDeviceSynchronize();
        // reset near len as near pile are distributed to buckets
        *near_len = 0;
      }

      // todo: possibly compact bucket

      // find the bucket with minimum size
      index_type min_bucket_size = 0, min_bucket = 0;
      found_non_empty_bucket = false;
      for (index_type i = 0; i < nBuckets; ++i) {
        if (buckets_len[i] > 0 &&
            (!found_non_empty_bucket || buckets_len[i] < min_bucket_size)) {
          min_bucket_size = buckets_len[i];
          min_bucket = i;
          found_non_empty_bucket = true;
        }
      }

      bucket_idx = min_bucket;
      ++iter;
    }

    // todo: compact far pile

    if (*far_len <= 0) {
      break;
    }

    // far split
    while (*near_len < 1) {
      delta += delta;
      far_split<<<(*far_len + 512 - 1) / 512, 512>>>(
          d_dists, delta, far, far_len, near, near_len, far2, far2_len);
      cudaDeviceSynchronize();

      index_type *tmp = far;
      far = far2;
      far2 = tmp;

      std::swap(far_len, far2_len);
      *far2_len = 0;
    }

    // split near into buckets
    buckets_split<<<(*near_len + 512 - 1) / 512, 512>>>(
        delta, d_dists, near, near_len, buckets, buckets_len, each_bucket_cap);
    cudaDeviceSynchronize();
    // reset near len as near pile are distributed to buckets
    *near_len = 0;

    index_type min_len = 0, min_len_bucket = 0;
    for (index_type i = 0; i < nBuckets; ++i) {
      if (buckets_len[i] > 0 &&
          (!found_non_empty_bucket || min_len > buckets_len[i])) {
        min_len = buckets_len[i];
        min_len_bucket = i;
        found_non_empty_bucket = true;
      }
    }

    bucket_idx = min_len_bucket;
  }

  end = getTimeStamp();
  double gpu_time = end - start;

  check_cuda(cudaFree(buckets));
  check_cuda(cudaFree(near));
  check_cuda(cudaFree(far));
  check_cuda(cudaFree(far2));
  check_cuda(cudaFree(buckets_len));
  check_cuda(cudaFree(near_len));
  check_cuda(cudaFree(far_len));
  check_cuda(cudaFree(far2_len));
  overhead += getTimeStamp() - end;

  cudaMemcpy(dists, d_dists, g.nnodes * sizeof(edge_data_type),
             cudaMemcpyDeviceToHost);

  check_cuda(cudaFree(d_dists));

  return {gpu_time, overhead};
}
