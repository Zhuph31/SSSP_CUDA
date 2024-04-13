#pragma once
#include "csr.h"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
template <int block_size>
__device__ void exclusive_scan(index_type *output, const index_type *input,
                               index_type n) {
  __shared__ index_type temp[block_size * 2];
  index_type thread_id = threadIdx.x;
  index_type ai = thread_id;
  index_type bi = thread_id + (n / 2);
  index_type bank_offset_a = CONFLICT_FREE_OFFSET(ai);
  index_type bank_offset_b = CONFLICT_FREE_OFFSET(bi);

  if (thread_id < n) {
    temp[ai + bank_offset_a] = input[ai];
    temp[bi + bank_offset_b] = input[bi];
  } else {
    temp[ai + bank_offset_a] = 0;
    temp[bi + bank_offset_b] = 0;
  }

  // up sweep
  index_type offset = 1;
  for (index_type d = block_size >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (thread_id < d) {
      index_type ai = offset * (2 * thread_id + 1) - 1;
      index_type bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thread_id == 0) {
    index_type target_val = 0;
    temp[block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1)] = target_val;
  }

  // down sweep
  for (index_type d = 1; d < block_size; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thread_id < d) {
      index_type ai = offset * (2 * thread_id + 1) - 1;
      index_type bi = offset * (2 * thread_id + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      index_type t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  __syncthreads();
  if (thread_id < n) {
    output[ai] = temp[ai + bank_offset_a];
    output[bi] = temp[bi + bank_offset_b];
  }
  __syncthreads();
}