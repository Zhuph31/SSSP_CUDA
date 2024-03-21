#ifndef WORKFRONT_SWEEP_H
#define WORKFRONT_SWEEP_H

#include "csr.h"

void workfront_sweep(CSRGraph& g, edge_data_type* dists);

#endif // WORKFRONT_SWEEP_H