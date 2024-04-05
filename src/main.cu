#include "common.cuh"
#include "csr.h"
#include "bellman-ford.cuh"
#include "workfront-sweep.cuh"
#include "nearfar.cuh"
#include "bucketing.cuh"
#include "dijkstra.h"


#include <math.h>



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




int main(int argc, char** argv) {
    CSRGraph g, gg;
    double start,end = 0;

    if (argc != 2){
        printf("usage program <dataset path>\n");
        return 1; 
    }

    g.read(argv[1]); 
    //g.read("inputs/rmat22.gr");
    //init_trivial_graph(g);

    std::vector<edge_data_type> out_cpu;

    start = getTimeStamp();
    dijkstra(g, out_cpu);
    end = getTimeStamp();
    double cpu_time = end - start;
    printf("CPU time: %f\n",cpu_time);


    edge_data_type* h_d = NULL;
    check_cuda(cudaMallocHost(&h_d, g.nnodes * sizeof(edge_data_type),cudaHostAllocWriteCombined));

    start = getTimeStamp();

    workfront_sweep(g, h_d);
    //nearfar(g,h_d);
    
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("Total GPU time: %f\n",gpu_time);

    compare(out_cpu,h_d);
    return 0;
}