#include "csr.h"
#include "utils.h"
#include <sys/time.h>
#include <thrust/sort.h>

const uint32_t nBuckets = 10;

__global__ void bucketing_iter(CSRGraph g, edge_data_type *d,
                               edge_data_type delta, index_type *bucket,
                               index_type bucket_len, index_type *near,
                               index_type *near_len, index_type *far,
                               index_type *far_len) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

  // only work on closest bucket pile
  if (index < bucket_len) {
    index_type s_idx = bucket[index];
    // printf("index %u, node:%u, edges:%u\n", index, s_idx,
    //        g.row_start[s_idx + 1]);
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
        // printf("updated approximation for node %u, %u vs %u\n", n, new_w,
        // nw);
        atomicMin(&d[n], new_w);

        // seperating into near and far pile
        index_type q_idx;
        if (new_w < delta) {
          q_idx = atomicAdd(near_len, 1);
          near[q_idx] = n;
          // printf("put node %u into near pile, idx:%u\n", n, q_idx);
        } else {
          // printf("far_len:%u\n", *far_len);
          q_idx = atomicAdd(far_len, 1); // ! could possibly cause far overflow
          // printf("far_len updated to be:%u\n", *far_len);
          far[q_idx] = n;
          // printf("put node %u into far pile, idx:%u\n", n, q_idx);
        }
      }
    }
  }
}

// split near pile into buckets
__global__ void buckets_split(edge_data_type delta, edge_data_type *d_d,
                              index_type *near, index_type *near_len,
                              index_type *buckets, index_type *buckets_len,
                              index_type each_bucket_cap) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
  if (index < *near_len) {
    index_type n_idx = near[index];
    edge_data_type w = d_d[n_idx];

    edge_data_type base_delta = delta / 10;
    edge_data_type bucket_idx = w / base_delta;
    if (bucket_idx > nBuckets - 1) {
      bucket_idx = nBuckets - 1;
    }

    // printf("buckets_split, index:%u, node_idx:%u, distance:%u,
    // bucket_idx:%u\n",
    //        index, n_idx, w, bucket_idx);
    index_type q_idx = atomicAdd(&buckets_len[bucket_idx], 1);
    index_type *bucket_start =
        buckets +
        bucket_idx * each_bucket_cap; // start of the corresponding bucket

    *(bucket_start + q_idx) = n_idx; // ! possible overflow
    // printf("Adding %d to bucket %u, q[%d]\n", n_idx, bucket_idx, q_idx);
  }
}

__global__ inline void
far_split(edge_data_type *d, edge_data_type delta, index_type *last_far_pile,
          index_type *last_far_len, index_type *new_near_pile,
          index_type *new_near_len, index_type *new_far_pile,
          index_type *new_far_len) {
  index_type index = threadIdx.x + (blockDim.x * blockIdx.x);
  // only split far pile into near and far
  if (index < *last_far_len) {
    index_type far_idx = last_far_pile[index];
    edge_data_type nw = d[far_idx];
    // printf("index %d, far_idx %d, distance %d\n", index, far_idx, nw);
    index_type q_idx;
    if (nw < delta) {
      q_idx = atomicAdd(new_near_len, 1);
      new_near_pile[q_idx] = far_idx;
    } else {
      q_idx = atomicAdd(new_far_len, 1);
      new_far_pile[q_idx] = far_idx;
    }
    // printf("Adding %d to q[%d]\n",n,q_idx);
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

  check_cuda(cudaMalloc(&buckets, bucket_total_cap * sizeof(index_type)));
  check_cuda(cudaMalloc(&near, g.nedges * sizeof(index_type)));
  check_cuda(cudaMalloc(
      &far,
      g.nedges * 2 *
          sizeof(index_type))); // ! how large does far need to avoid overflow ?
  check_cuda(cudaMalloc(
      &far2,
      g.nedges * 2 *
          sizeof(index_type))); // ! how large does far need to avoid overflow ?
  printf("far size:%u\n", g.nedges * 2);

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

  printf("weight %d \n", g.getWeight(0, 0));
  start = getTimeStamp();

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
      printf("Iter %d, far batch %d, delta %d, current bucket:%u, current "
             "bucket size:%u\n",
             iter, *far_len, delta, bucket_idx, buckets_len[bucket_idx]);

      index_type current_bucket_len = buckets_len[bucket_idx];

      index_type *current_bucket =
          buckets +
          bucket_idx * each_bucket_cap; // calculate the start of current bucket

      bucketing_iter<<<(current_bucket_len + 512 - 1) / 512, 512>>>(
          d_g, d_d, delta, current_bucket, current_bucket_len, near, near_len,
          far, far_len);
      cudaDeviceSynchronize();
      // getchar();

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("\033[0;31mBucketing iteration error: %s\033[0m\n",
               cudaGetErrorString(err));
        exit(1);
      }

      // set current bucket size to 0 as they are all processed
      buckets_len[bucket_idx] = 0;
      // printf("finished bucketing iter \n");

      // todo: possibly compact near & near_dis?
      // ? why do we need to sort here?

      // printf("sorted near pile\n");
      // printf("near len:%u\n", *near_len);

      // split near pile into buckets only when near pile is not empty
      if (*near_len > 0) {
        index_type griddim = (*near_len + 512 - 1) / 512;
        // printf("calling buckets split, with griddim:%u\n", griddim);
        buckets_split<<<griddim, 512>>>(delta, d_d, near, near_len, buckets,
                                        buckets_len, each_bucket_cap);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("\033[0;31mBucket split error: %s\033[0m\n",
                 cudaGetErrorString(err));
          exit(1);
        }
        cudaDeviceSynchronize();
        // reset near len as near pile are distributed to buckets
        *near_len = 0;

        // printf("splitted near pile into buckets\n");
      }

      // debug bucket split result
      // for (index_type i = 0; i < nBuckets; ++i) {
      //   if (buckets_len[i] <= 0) {
      //     printf("bucket %u empty\n", i);
      //   } else {
      //     printf("bucket %u size %u, ", i, buckets_len[i]);
      //     for (index_type j = 0; j < buckets_len[i]; ++j) {
      //       printf("%u, ", buckets[i * each_bucket_cap + j]);
      //     }
      //     printf("\n");
      //   }
      // }

      // todo: possibly compact bucket

      // find the bucket with minimum size
      index_type min_bucket_size = std::numeric_limits<index_type>::max(),
                 min_bucket = 0;
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
      // printf("bucket for next iteration:%u\n", bucket_idx);
      // for (index_type i = 0; i < nBuckets; ++i) {
      //   printf("bucket %u content:\n", i);
      //   index_type *each_bucket = buckets + i * each_bucket_cap;
      //   for (index_type j = 0; j < buckets_len[i]; ++j) {
      //     printf("%u,", each_bucket[j]);
      //   }
      //   printf("\n");
      // }
    }

    // printf("all buckets are empty, dealing with far\n");

    // todo: compact far pile

    if (*far_len <= 0) {
      // printf("done, break\n");
      break;
    }

    // far split
    while (*near_len < 1) {
      delta += delta;

      // printf("before far split, far len:%u\n", *far_len);

      // printf("debug far:\n");
      // for (index_type i = 0; i < *far_len; ++i) {
      //   printf("%u(%u), ", far[i], d_d[far[i]]);
      // }

      far_split<<<(*far_len + 512 - 1) / 512, 512>>>(
          d_d, delta, far, far_len, near, near_len, far2, far2_len);
      cudaDeviceSynchronize();
      // printf("near size:%u, far size:%u, far2 size:%u, delta:%u\n",
      // *near_len,
      //        *far_len, *far2_len, delta);

      index_type *tmp = far;
      far = far2;
      far2 = tmp;

      std::swap(far_len, far2_len);
      *far2_len = 0;
      // printf("after swap, near size:%u, far size:%u, far2 size:%u\n",
      // *near_len,
      //        *far_len, *far2_len);

      // printf("debug near:\n");
      // for (index_type i = 0; i < *near_len; ++i) {
      //   printf("%u(%u), ", near[i], d_d[near[i]]);
      // }

      // getchar();
    }

    // printf("far split finish, far:%u, near:%u\n", *far_len, *near_len);

    // split near into buckets
    // ? evenly split based on delta ?
    // printf("out loop before buckets split, delta:%u\n", delta);
    buckets_split<<<(*near_len + 512 - 1) / 512, 512>>>(
        delta, d_d, near, near_len, buckets, buckets_len, each_bucket_cap);
    cudaDeviceSynchronize();
    // reset near len as near pile are distributed to buckets
    *near_len = 0;

    index_type min_len = 0, min_len_bucket = 0;
    for (index_type i = 0; i < nBuckets; ++i) {
      // printf("bucket %u len %u\n", i, buckets_len[i]);
      if (buckets_len[i] > 0 &&
          (!found_non_empty_bucket || min_len > buckets_len[i])) {
        min_len = buckets_len[i];
        min_len_bucket = i;
        found_non_empty_bucket = true;
      }
    }

    bucket_idx = min_len_bucket;
    // getchar();
  }

  end = getTimeStamp();
  double gpu_time = end - start;
  printf("GPU time: %f\n", gpu_time);

  cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type),
             cudaMemcpyDeviceToHost);
}
