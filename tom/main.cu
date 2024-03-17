#include "csr.h"
#include <vector>
#include <queue>
#include <tuple>
#include <iostream>
#include <sys/time.h>
#include <math.h>

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}


void dijkstra(const CSRGraph& g, std::vector<edge_data_type>& dists) {
    typedef std::tuple<index_type, edge_data_type> Node;

    auto cmp = [](Node left, Node right) { return (std::get<1>(left)) > (std::get<1>(right)); };
    std::priority_queue<Node,std::vector<Node>,decltype(cmp)> pq(cmp);

    pq.push(Node{0,0});

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

__global__ const edge_data_type MAX_VAL = UINT_MAX;
__global__ void bf_iter(CSRGraph g, edge_data_type* last_d, edge_data_type* new_d) {
    index_type index = threadIdx.x + (blockDim.x * blockIdx.x);

    if (index < g.nnodes) {
        // Only process nodes that changed last time
        if (last_d[index] == new_d[index]) {
            return;
        }
        atomicMin(&new_d[index], last_d[index]);

        // somewhat baed on https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b
        for (int j = g.row_start[index]; j < g.row_start[index + 1]; j++) {
            edge_data_type w = last_d[index];
            edge_data_type ew = g.edge_data[j];
            index_type n = g.edge_dst[j];
            edge_data_type nw = last_d[n];
            edge_data_type new_w = ew + w;
            // Check if the distance is already set to max then just take the max since,
            if (w >= MAX_VAL){
                new_w = MAX_VAL;
            }

            if (new_w < nw) {
                atomicMin(&new_d[n],new_w);
            }
        }
    }
}

bool compare(std::vector<edge_data_type>& cpu, edge_data_type* gpu) {

    for (int i = 0; i < cpu.size(); i++) {
        if (cpu[i] != gpu[i]) {
            printf("Wrong at %d: %d!=%d\n",i,cpu[i],gpu[i]);
            return false;
        }
    }

    printf("Match!\n");
    return true;
}

void init_trivial(CSRGraph& g) {
    g.nnodes = 6;
    g.nedges = 16;
    g.row_start = (index_type *)malloc((g.nnodes+1)*sizeof(index_type));
    g.edge_dst = (index_type *)malloc(g.nedges*sizeof(edge_data_type));
    g.edge_data = (edge_data_type *)malloc(g.nedges*sizeof(edge_data_type));
    g.node_data = (node_data_type *)malloc(g.nnodes*sizeof(edge_data_type));

    g.row_start[0] = 0;
    g.row_start[1] = 2;
    g.row_start[2] = 5;
    g.row_start[3] = 9;
    g.row_start[4] = 11;
    g.row_start[5] = 13;
    g.row_start[6] = 16;

    g.edge_dst[0] = 1;
    g.edge_dst[1] = 2;

    g.edge_dst[2] = 0;
    g.edge_dst[3] = 2;
    g.edge_dst[4] = 3;

    g.edge_dst[5] = 0;
    g.edge_dst[6] = 1;
    g.edge_dst[7] = 4;
    g.edge_dst[8] = 5;

    g.edge_dst[9] = 1;
    g.edge_dst[10] = 5;

    g.edge_dst[11] = 2;
    g.edge_dst[12] = 5;

    g.edge_dst[13] = 2;
    g.edge_dst[14] = 3;
    g.edge_dst[15] = 4;

    g.edge_data[0] = 2;
    g.edge_data[1] = 4;

    g.edge_data[2] = 2;
    g.edge_data[3] = 1;
    g.edge_data[4] = 6;

    g.edge_data[5] = 4;
    g.edge_data[6] = 1;
    g.edge_data[7] = 2;
    g.edge_data[8] = 3;

    g.edge_data[9] = 6;
    g.edge_data[10] = 5;

    g.edge_data[11] = 2;
    g.edge_data[12] = 4;

    g.edge_data[13] = 3;
    g.edge_data[14] = 5;
    g.edge_data[15] = 4;
}


void bf_impl(CSRGraph& g, edge_data_type* dists) {
    double start,end = 0;
    CSRGraph d_g;
    g.copy_to_gpu(d_g);
    edge_data_type* d_d, *d_d1, *d_d2  = NULL;
    check_cuda(cudaMalloc(&d_d1, g.nnodes * sizeof(edge_data_type)));
    check_cuda(cudaMalloc(&d_d2, g.nnodes * sizeof(edge_data_type)));
    // Initialize for source node = 0. Otherwise need to change this
    check_cuda(cudaMemset(&d_d1[1], 0xFF,  (g.nnodes-1) * sizeof(edge_data_type)));
    check_cuda(cudaMemset(&d_d2[0], 0xFF,  (g.nnodes) * sizeof(edge_data_type)));

    start = getTimeStamp();
    for (int i = 0; i < g.nnodes -1; i++) {
        // printf("Iter %d\n",i);
        edge_data_type* tmp = d_d1;
        d_d1 = d_d2;
        d_d2 = tmp;
 
        bf_iter<<<(g.nnodes + 512 - 1) / 512,512>>>(d_g, d_d1, d_d2);
    }
    d_d = d_d2;
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("GPU time: %f\n",gpu_time);

    cudaMemcpy(dists, d_d, g.nnodes * sizeof(edge_data_type), cudaMemcpyDeviceToHost);
}





int main(int argc, char** argv) {
    CSRGraph g, gg;
    double start,end = 0;
    
    g.read("inputs/rmat20.gr");
    // init_trivial(g);


    std::vector<edge_data_type> out_cpu;

    start = getTimeStamp();
    dijkstra(g, out_cpu);
    end = getTimeStamp();
    double cpu_time = end - start;
    printf("CPU time: %f\n",cpu_time);


    edge_data_type* h_d = NULL;
    check_cuda(cudaMallocHost(&h_d, g.nnodes * sizeof(edge_data_type)));

    start = getTimeStamp();
    bf_impl(g, h_d);
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("Total GPU time: %f\n",gpu_time);

    compare(out_cpu,h_d);
    return 0;
}