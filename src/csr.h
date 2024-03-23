/*
   csr_graph.h

   Implements a CSR Graph. Part of the GGC source code. 
   Interface derived from LonestarGPU.

   Copyright (C) 2014--2016, The University of Texas at Austin

   See LICENSE.TXT for copyright license.

   Author: Sreepathi Pai <sreepai@ices.utexas.edu> 
*/

#ifndef LSG_CSR_GRAPH
#define LSG_CSR_GRAPH

#include "common.cuh"

#include <assert.h>


// Adapted from LSG CSRGraph.h

// very simple implementation
struct CSRGraph {
  unsigned read(char file[]);
  void copy_to_gpu(struct CSRGraph &copygraph);
  void copy_to_cpu(struct CSRGraph &copygraph);

  CSRGraph();

  unsigned init();
  unsigned allocOnHost();
  unsigned allocOnDevice();
  void progressPrint(unsigned maxii, unsigned ii);
  unsigned readFromGR(char file[]);

  unsigned deallocOnHost();
  unsigned deallocOnDevice();
  void dealloc();

  __device__ __host__ bool valid_node(index_type node) {
    return (node < nnodes);
  }

  __device__ __host__ bool valid_edge(index_type edge) {
    return (edge < nedges);
  }

  __device__ __host__ index_type getOutDegree(unsigned src) {
    assert(src < nnodes);
    return row_start[src+1] - row_start[src];
  };

  __device__ __host__ index_type getDestination(unsigned src, unsigned edge) {
      assert(src < nnodes);
      assert(edge < getOutDegree(src));

      index_type abs_edge = row_start[src] + edge;
      assert(abs_edge < nedges);
      
      return edge_dst[abs_edge];
  };

  __device__ __host__ index_type getAbsDestination(unsigned abs_edge) {
    assert(abs_edge < nedges);
  
    return edge_dst[abs_edge];
  };

  __device__ __host__ index_type getFirstEdge(unsigned src) {
    assert(src <= nnodes); // <= is okay
    return row_start[src];
  };

  __device__ __host__ edge_data_type    getWeight(unsigned src, unsigned edge) {
  assert(src < nnodes);
  assert(edge < getOutDegree(src));

  index_type abs_edge = row_start[src] + edge;
  assert(abs_edge < nedges);
  
  return edge_data[abs_edge];
    
  };

  __device__ __host__ edge_data_type    getAbsWeight(unsigned abs_edge) {
  assert(abs_edge < nedges);
  
  return edge_data[abs_edge];
    
  };

  index_type nnodes, nedges;
  index_type *row_start; // row_start[node] points into edge_dst, node starts at 0, row_start[nnodes] = nedges
  index_type *edge_dst;
  edge_data_type *edge_data;
  node_data_type *node_data; 
  bool device_graph;

};

void init_trivial_graph(CSRGraph& g);

#endif// LSG_CSR_GRAPH
