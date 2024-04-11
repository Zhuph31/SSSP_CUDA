#include "common.cuh"
#include "csr.h"
#include "bellman-ford.cuh"
#include "workfront-sweep.cuh"
#include "nearfar.cuh"
#include "bucketing.cuh"
#include "dijkstra.h"


#include <math.h>
#include <filesystem>
#include <cstdlib>
#include <fstream>





int main(int argc, char** argv) {
    CSRGraph g, gg;
    double start,end = 0;

    if (argc != 2){
        printf("usage program <dataset path>\n");
        return 1; 
    }

    assert(std::filesystem::exists("src/Parallel-SSSP/sssp"));

    g.read(argv[1]); 
    //g.read("inputs/rmat22.gr");
    // init_trivial_graph(g);

    std::vector<edge_data_type> out_cpu;

    start = getTimeStamp();
    dijkstra(g, out_cpu, 0);
    end = getTimeStamp();
    double cpu_time = end - start;
    printf("CPU time: %f\n",cpu_time);

    double cpu_mult_time;
    system("rm sssp.tsv -f");
    std::string cmd = "src/Parallel-SSSP/sssp -w -a rho-stepping -i "+std::string(argv[1])+">/dev/null";
    system(cmd.c_str());
    std::ifstream ifs("sssp.tsv");
    ifs >> cpu_mult_time;
    ifs.close();
    printf("Rho-stepping time: %f\n", cpu_mult_time);
    system("rm sssp.tsv -f");

    system("rm sssp.tsv -f");
    cmd = "src/Parallel-SSSP/sssp -w -a delta-stepping -i "+std::string(argv[1])+">/dev/null";
    system(cmd.c_str());
    ifs.open("sssp.tsv");
    ifs >> cpu_mult_time;
    ifs.close();
    printf("Delta-stepping time: %f\n",cpu_mult_time);
    system("rm sssp.tsv -f");

    edge_data_type* h_d = NULL;
    check_cuda(cudaMallocHost(&h_d, g.nnodes * sizeof(edge_data_type),cudaHostAllocWriteCombined));

    start = getTimeStamp();

    // workfront_sweep(g, h_d, 0);
    workfront_sweep_evaluation(g, h_d, 0, out_cpu.data());
    //nearfar(g,h_d);
    
    end = getTimeStamp();
    double gpu_time = end - start;
    printf("Total GPU time: %f\n",gpu_time);


    compare(out_cpu.data(),h_d, out_cpu.size());
    // for (int i = 0; i < 40; i++) {
    //     printf("%d ",h_d[i]);
    // }
    // printf("\n");
    return 0;
}