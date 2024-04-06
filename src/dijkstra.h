#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include <vector>
#include "csr.h"

void dijkstra(const CSRGraph& g, std::vector<edge_data_type>& dists, index_type source);

#endif //DIJKSTRA_H