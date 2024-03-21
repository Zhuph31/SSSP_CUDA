#ifndef BELLMAN_FORD_H
#define BELLMAN_FORD_H

#include "common.cuh"
#include "csr.h"

void bellman_ford(CSRGraph& g, edge_data_type* dists);

#endif // BELLMAN_FORD_H
