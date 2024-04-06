#include "dijkstra.h"
#include <queue>
#include <tuple>
#include <iostream>
#include <math.h>
#include <climits>


void dijkstra(const CSRGraph& g, std::vector<edge_data_type>& dists, index_type source) {
    typedef std::tuple<index_type, edge_data_type> Node;

    auto cmp = [](Node left, Node right) { return (std::get<1>(left)) > (std::get<1>(right)); };
    std::priority_queue<Node,std::vector<Node>,decltype(cmp)> pq(cmp);

    pq.push(Node{source,0});

    dists.resize(g.nnodes, UINT_MAX);
    std::vector<bool> explored(g.nnodes);


    while (!pq.empty()) {
        auto node = pq.top();
        pq.pop();
        index_type s = std::get<0>(node);
        edge_data_type w = std::get<1>(node);

        if (explored[s]) {
            continue;
        }
        explored[s] = true;
        dists[s] = w;

        index_type start_idx = g.row_start[s];
        index_type end_idx = g.row_start[s+1];

        for (index_type idx = start_idx; idx < end_idx; idx++) {
            index_type n = g.edge_dst[idx];
            if (explored[n]) {
                continue;
            }
            edge_data_type ew = g.edge_data[idx];
            edge_data_type new_w = w + ew;
            if (new_w < dists[n]) {
                dists[n] = new_w;
                pq.push(Node{n, new_w});
            }
        }
    }

}
