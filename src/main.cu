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



int main(int argc, char** argv) {
    CSRGraph g, gg;
    double start,end = 0;

    if (argc != 2){
        printf("usage program <dataset path>\n");
        return 1; 
    }

    g.read(argv[1]); 
    //g.read("inputs/rmat22.gr");
    // init_trivial_graph(g);
    // g.writeToCSR("out.csr");


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
    // nearfar(g,h_d);
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("Total GPU time: %f\n",gpu_time);


    compare(out_cpu,h_d);
    for (int i = 0; i < 40; i++) {
        printf("%d ",h_d[i]);
    }
    printf("\n");
    return 0;
}