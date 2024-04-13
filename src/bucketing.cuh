#pragma once

#include "csr.h"
#include "utils.cuh"

TimeCost bucketing(CSRGraph &g, edge_data_type *dists);