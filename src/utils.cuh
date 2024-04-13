#pragma once
#include "csr.h"

struct TimeCost {
  double gpu_time;
  double overhead;
  TimeCost() = default;
  TimeCost(double g, double o) : gpu_time(g), overhead(o) {}
};

inline edge_data_type calculate_delta(CSRGraph g) {
  unsigned long ew = 0;
  for (int i = 0; i < g.nedges; i++) {
    ew += g.edge_data[i];
  }
  ew = ew / g.nedges;
  // calculate average edge degree as total # of edges / total # of nodes
  edge_data_type d = (g.nedges / g.nnodes);
  return 32 * ew / d;
}
