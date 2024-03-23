#include "csr.h"
#include <sys/time.h>

inline double getTimeStamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

inline edge_data_type calculate_delta(CSRGraph g) {
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

const edge_data_type MAX_VAL = UINT_MAX;
