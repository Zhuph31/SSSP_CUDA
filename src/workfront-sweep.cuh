#ifndef WORKFRONT_SWEEP_H
#define WORKFRONT_SWEEP_H

#include "csr.h"

void workfront_sweep(CSRGraph& g, edge_data_type* dists, index_type source);

void workfront_sweep_evaluation(CSRGraph& g, edge_data_type* dists, index_type source, edge_data_type* cpu);

#endif // WORKFRONT_SWEEP_H