#ifndef NEARFAR_H
#define NEARFAR_H

#include "csr.h"
#include "utils.cuh"

TimeCost nearfar(CSRGraph &g, edge_data_type *dists);

#endif // WORKFRONT_SWEEP_H